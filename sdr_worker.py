import time
import queue as _q
import threading
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

import dsp_utils

try:
    import adi
    _PLUTO_OK = True
except ImportError:
    _PLUTO_OK = False


class SDRWorker(QThread):
    """Acquisition + DSP thread. Emits FFT data and demodulated audio."""

    fft_ready    = pyqtSignal(np.ndarray, np.ndarray)  # (freqs_hz, power_db)
    audio_ready  = pyqtSignal(np.ndarray)               # float32 audio chunk
    decoded_data = pyqtSignal(bytes)
    status_msg   = pyqtSignal(str)
    error_msg    = pyqtSignal(str)
    connected    = pyqtSignal(bool)
    tx_spectrum_ready = pyqtSignal(np.ndarray, np.ndarray)  # (freqs_hz, power_db) for TX monitoring

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sdr      = None
        self._running  = False
        self._sim      = not _PLUTO_OK
        self._mutex    = QMutex()
        self._pending  = {}
        self._tx_waveform = np.zeros(0, dtype=np.complex64)
        self._tx_pos      = 0
        self._tx_thread   = None
        self._afsk_decoder = dsp_utils.AFSKDecoder(dsp_utils.AUDIO_RATE)

        # SDR parameters
        self.sample_rate  = int(2e6)
        self.center_freq  = int(100e6)
        self.gain         = 40.0
        self.gain_mode    = "manual"
        self.buffer_size  = 1024 * 8
        self.pluto_uri    = "ip:192.168.2.1"

        # Demodulation parameters
        self.rx_offset    = 0.0          # Hz offset within passband
        self.demod_mode   = "nfm"
        self.channel_bw   = 12.5e3
        self.audio_on     = True

        # TX streaming parameters
        self._tx_streaming = False
        self._tx_audio_queue: _q.Queue = _q.Queue(maxsize=8)
        self._tx_proc_thread: threading.Thread | None = None
        self._tx_modulation = "nbfm"
        self._tx_rf_offset = 0.0
        self._tx_tx_gain = 0.8
        self._fm_mod = dsp_utils.FMModulator(deviation=5e3, sample_rate=dsp_utils.AUDIO_RATE)

        # Simulation state
        self._sim_t = 0.0

    # ── connection ──────────────────────────────────────────────────────────

    @property
    def simulation_mode(self):
        return self._sim

    def try_connect(self, uri: str = None):
        if uri:
            self.pluto_uri = uri
        if not _PLUTO_OK:
            self._sim = True
            self.status_msg.emit("Simulation mode (pyadi-iio not installed)")
            self.connected.emit(False)
            return False
        try:
            sdr = adi.Pluto(self.pluto_uri)
            sdr.sample_rate           = self.sample_rate
            sdr.rx_lo                 = self.center_freq
            sdr.rx_rf_bandwidth       = self.sample_rate
            sdr.rx_buffer_size        = self.buffer_size
            sdr.gain_control_mode_chan0 = self.gain_mode
            if self.gain_mode == "manual":
                sdr.rx_hardwaregain_chan0 = self.gain
            self._sdr = sdr
            self._sim = False
            self._setup_tx()
            self.status_msg.emit(f"Connected: {self.pluto_uri}")
            self.connected.emit(True)
            return True
        except Exception as exc:
            self._sim = True
            self.status_msg.emit(f"Simulation (no device: {exc})")
            self.connected.emit(False)
            return False

    def _setup_tx(self):
        if self._sdr is None:
            return
        try:
            self._sdr.tx_lo = self.center_freq
        except Exception:
            pass
        try:
            self._sdr.tx_rf_bandwidth = self.sample_rate
        except Exception:
            pass
        try:
            self._sdr.tx_buffer_size = self.buffer_size
        except Exception:
            pass
        if self.gain_mode == "manual":
            try:
                self._sdr.tx_hardwaregain_chan0 = self.gain
            except Exception:
                pass

    def _hardware_tx_worker(self, waveform: np.ndarray):
        try:
            self._sdr.tx(waveform.astype(np.complex64))
            self.status_msg.emit("Hardware message transmitted")
        except Exception as exc:
            self.error_msg.emit(f"TX error: {exc}")

    # ── parameter setters (thread-safe) ─────────────────────────────────────

    def set_center_freq(self, freq: int):
        with QMutexLocker(self._mutex):
            self._pending['center_freq'] = int(freq)

    def set_gain(self, gain: float, mode: str = "manual"):
        with QMutexLocker(self._mutex):
            self._pending['gain']      = gain
            self._pending['gain_mode'] = mode

    def set_sample_rate(self, rate: int):
        with QMutexLocker(self._mutex):
            self._pending['sample_rate'] = int(rate)

    def set_demod(self, mode: str, offset: float, bw: float):
        with QMutexLocker(self._mutex):
            self._pending['demod_mode'] = mode
            self._pending['rx_offset']  = offset
            self._pending['channel_bw'] = bw

    def set_audio(self, enabled: bool):
        self.audio_on = enabled

    def stop(self):
        self._running = False
        self.wait(2000)
    # ── TX streaming (microphone → modulate → resample → SDR) ─────────────────

    def start_tx_stream(self, modulation: str = "nbfm", rf_offset: float = 0.0):
        """Start streaming TX from microphone. Spawns a dedicated TX thread."""
        with QMutexLocker(self._mutex):
            self._tx_streaming = True
            self._tx_modulation = modulation.lower()
            self._tx_rf_offset = rf_offset

        # Reconfigure FM modulator for selected mode
        deviation = 75e3 if modulation.lower() == 'wfm' else 5e3
        self._fm_mod = dsp_utils.FMModulator(deviation=deviation,
                                              sample_rate=dsp_utils.AUDIO_RATE)
        # Flush any stale audio in the queue
        while not self._tx_audio_queue.empty():
            try:
                self._tx_audio_queue.get_nowait()
            except _q.Empty:
                break

        self._tx_proc_thread = threading.Thread(
            target=self._tx_stream_loop, daemon=True
        )
        self._tx_proc_thread.start()
        self.status_msg.emit(f"TX streaming started ({modulation.upper()})")

    def stop_tx_stream(self):
        """Stop streaming TX and join the TX thread."""
        with QMutexLocker(self._mutex):
            self._tx_streaming = False
        # Unblock the thread if it is waiting on an empty queue
        self._tx_audio_queue.put(None)
        if self._tx_proc_thread is not None:
            self._tx_proc_thread.join(timeout=2.0)
            self._tx_proc_thread = None
        self.status_msg.emit("TX streaming stopped")

    def enqueue_tx_audio(self, audio: np.ndarray):
        """Called from the mic worker signal — drop the chunk if the queue is full."""
        if not self._tx_streaming:
            return
        try:
            self._tx_audio_queue.put_nowait(audio.astype(np.float32))
        except _q.Full:
            pass  # prefer dropping over blocking the GUI thread

    def _tx_stream_loop(self):
        """Dedicated TX thread: dequeue audio chunks and transmit."""
        while True:
            try:
                audio = self._tx_audio_queue.get(timeout=0.05)
            except _q.Empty:
                if not self._tx_streaming:
                    break
                continue
            if audio is None:   # sentinel — stop
                break

            # Snapshot mutable params without holding the lock during processing
            with QMutexLocker(self._mutex):
                modulation  = self._tx_modulation
                rf_offset   = self._tx_rf_offset
                tx_gain     = self._tx_tx_gain
                sample_rate = self.sample_rate
                sim         = self._sim

            self._process_tx_chunk(audio, modulation, rf_offset, tx_gain,
                                   sample_rate, sim)

    def _process_tx_chunk(self, audio: np.ndarray, modulation: str,
                          rf_offset: float, tx_gain: float,
                          sample_rate: float, sim: bool):
        """Modulate one audio chunk and push it to the SDR hardware."""
        try:
            # Step 1 — Modulate at audio rate (stateful FM for phase continuity)
            if modulation == 'nbfm':
                iq = self._fm_mod.modulate(dsp_utils._pre_emphasis(audio, dsp_utils.AUDIO_RATE))
            elif modulation == 'wfm':
                iq = self._fm_mod.modulate(audio)
            elif modulation == 'am':
                iq = dsp_utils.modulate_am(audio, dsp_utils.AUDIO_RATE)
            else:
                return

            # Step 2 — Rational Resampler: audio rate → SDR sample rate
            iq = dsp_utils.resample_iq(iq, dsp_utils.AUDIO_RATE, int(sample_rate))

            # Step 3 — Frequency shift for RF offset (at full SDR rate)
            if rf_offset != 0.0:
                t = np.arange(len(iq), dtype=np.float64) / sample_rate
                iq = (iq * np.exp(2j * np.pi * rf_offset * t)).astype(np.complex64)

            # Step 4 — TX gain scaling
            iq = (iq * tx_gain).astype(np.complex64)

            # Step 5 — Emit spectrum for the QT GUI Frequency Sink
            freqs, pdb = dsp_utils.compute_fft(iq, sample_rate)
            self.tx_spectrum_ready.emit(freqs, pdb)

            # Step 6 — SDR Sink
            if not sim and self._sdr is not None:
                self._sdr.tx(iq)

        except Exception as exc:
            self.error_msg.emit(f"TX error: {exc}")
    # ── thread loop ─────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        while self._running:
            self._apply_pending()
            try:
                samples = self._acquire()
                freqs, pdb = dsp_utils.compute_fft(samples, self.sample_rate)
                self.fft_ready.emit(freqs, pdb)
                if self.audio_on:
                    audio = dsp_utils.tune_and_demodulate(
                        samples, self.sample_rate,
                        self.rx_offset, self.channel_bw,
                        self.demod_mode,
                    )
                    if len(audio):
                        decoded = self._afsk_decoder.feed(audio)
                        for packet in decoded:
                            self.decoded_data.emit(packet)
                        self.audio_ready.emit(audio)
            except Exception as exc:
                self.error_msg.emit(str(exc))
                time.sleep(0.05)

    def _apply_pending(self):
        with QMutexLocker(self._mutex):
            ch = dict(self._pending)
            self._pending.clear()

        for k, v in ch.items():
            if k == 'center_freq':
                self.center_freq = v
                if self._sdr and not self._sim:
                    try:
                        self._sdr.rx_lo = v
                        self._sdr.tx_lo = v
                    except Exception as e: self.error_msg.emit(f"Freq: {e}")
            elif k == 'gain':
                self.gain = v
                if self._sdr and not self._sim and self.gain_mode == "manual":
                    try:
                        self._sdr.rx_hardwaregain_chan0 = v
                        self._sdr.tx_hardwaregain_chan0 = v
                    except Exception as e: self.error_msg.emit(f"Gain: {e}")
            elif k == 'gain_mode':
                self.gain_mode = v
                if self._sdr and not self._sim:
                    try:
                        self._sdr.gain_control_mode_chan0 = v
                        self._sdr.tx_gain_control_mode_chan0 = v
                    except Exception as e:
                        try:    self._sdr.gain_control_mode_chan0 = v
                        except Exception as e2: self.error_msg.emit(f"GainMode: {e2}")
            elif k == 'sample_rate':
                self.sample_rate = v
                if self._sdr and not self._sim:
                    try:
                        self._sdr.sample_rate       = v
                        self._sdr.rx_rf_bandwidth   = v
                        self._sdr.tx_rf_bandwidth   = v
                    except Exception as e: self.error_msg.emit(f"SRate: {e}")
            elif k == 'demod_mode': self.demod_mode = v
            elif k == 'rx_offset':  self.rx_offset  = v
            elif k == 'channel_bw': self.channel_bw = v

    def send_text(self, text: str):
        self.send_data(text.encode('utf-8'))

    def send_data(self, payload: bytes):
        if len(payload) > dsp_utils.AFSK_MAX_PAYLOAD:
            raise ValueError(f"Message too large: {len(payload)} bytes (max {dsp_utils.AFSK_MAX_PAYLOAD})")

        audio = dsp_utils.encode_afsk_bytes(payload, dsp_utils.AUDIO_RATE)
        waveform = dsp_utils.afsk_audio_to_fm(audio, self.sample_rate)
        self._tx_waveform = waveform
        self._tx_pos = 0

        if self._sim:
            self.status_msg.emit(f"Queued {len(payload)}-byte message for simulation TX")
            return

        if self._sdr is None or not hasattr(self._sdr, 'tx'):
            self.status_msg.emit("Connected device does not expose TX interface")
            return

        if self._tx_thread is not None and self._tx_thread.is_alive():
            self.status_msg.emit("Previous TX still in progress")
            return

        self._tx_thread = threading.Thread(
            target=self._hardware_tx_worker,
            args=(waveform,),
            daemon=True,
        )
        self._tx_thread.start()
        self.status_msg.emit(f"Transmitting {len(payload)} bytes on hardware")

    def _acquire(self) -> np.ndarray:
        if self._sim:
            return self._simulate()
        raw = self._sdr.rx()
        return np.array(raw, dtype=np.complex64) / 2**14

    def _simulate(self) -> np.ndarray:
        n  = self.buffer_size
        if self._tx_pos < len(self._tx_waveform):
            end = min(self._tx_pos + n, len(self._tx_waveform))
            sig = self._tx_waveform[self._tx_pos:end]
            self._tx_pos = end
            if self._tx_pos >= len(self._tx_waveform):
                self._tx_waveform = np.zeros(0, dtype=np.complex64)
                self.status_msg.emit("Message transmission finished")
            if len(sig) < n:
                remainder = self._simulate() if self._tx_pos >= len(self._tx_waveform) else np.zeros(n - len(sig), dtype=np.complex64)
                return np.concatenate((sig, remainder))
            return sig
        sr = self.sample_rate
        t  = np.arange(n, dtype=np.float64) / sr + self._sim_t
        self._sim_t += n / sr

        sig = np.zeros(n, dtype=np.complex128)

        # Narrow FM at center (0 Hz offset)
        mod = np.cumsum(np.sin(2 * np.pi * 800.0 * t)) / sr
        sig += 0.6 * np.exp(2j * np.pi * (0.0 * t + 25e3 * mod))

        # CW tone +400 kHz above center
        sig += 0.30 * np.exp(2j * np.pi * 400e3 * t)

        # AM signal −600 kHz below center
        env  = 1 + 0.8 * np.sin(2 * np.pi * 1500.0 * t)
        sig += 0.25 * env * np.exp(2j * np.pi * -600e3 * t)

        # Wide FM +700 kHz above center
        mod2 = np.cumsum(np.sin(2 * np.pi * 440.0 * t)) / sr
        sig += 0.45 * np.exp(2j * np.pi * (700e3 * t + 75e3 * mod2))

        # Noise
        sig += (np.random.randn(n) + 1j * np.random.randn(n)) * 0.010

        time.sleep(n / sr * 0.85)   # pace to near real-time
        return sig.astype(np.complex64)
