import time
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

    fft_ready   = pyqtSignal(np.ndarray, np.ndarray)  # (freqs_hz, power_db)
    audio_ready = pyqtSignal(np.ndarray)               # float32 audio chunk
    status_msg  = pyqtSignal(str)
    error_msg   = pyqtSignal(str)
    connected   = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sdr     = None
        self._running = False
        self._sim     = not _PLUTO_OK
        self._mutex   = QMutex()
        self._pending = {}

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
            self.status_msg.emit(f"Connected: {self.pluto_uri}")
            self.connected.emit(True)
            return True
        except Exception as exc:
            self._sim = True
            self.status_msg.emit(f"Simulation (no device: {exc})")
            self.connected.emit(False)
            return False

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
                    try:    self._sdr.rx_lo = v
                    except Exception as e: self.error_msg.emit(f"Freq: {e}")
            elif k == 'gain':
                self.gain = v
                if self._sdr and not self._sim and self.gain_mode == "manual":
                    try:    self._sdr.rx_hardwaregain_chan0 = v
                    except Exception as e: self.error_msg.emit(f"Gain: {e}")
            elif k == 'gain_mode':
                self.gain_mode = v
                if self._sdr and not self._sim:
                    try:    self._sdr.gain_control_mode_chan0 = v
                    except Exception as e: self.error_msg.emit(f"GainMode: {e}")
            elif k == 'sample_rate':
                self.sample_rate = v
                if self._sdr and not self._sim:
                    try:
                        self._sdr.sample_rate       = v
                        self._sdr.rx_rf_bandwidth   = v
                    except Exception as e: self.error_msg.emit(f"SRate: {e}")
            elif k == 'demod_mode': self.demod_mode = v
            elif k == 'rx_offset':  self.rx_offset  = v
            elif k == 'channel_bw': self.channel_bw = v

    def _acquire(self) -> np.ndarray:
        if self._sim:
            return self._simulate()
        raw = self._sdr.rx()
        return np.array(raw, dtype=np.complex64) / 2**14

    def _simulate(self) -> np.ndarray:
        n  = self.buffer_size
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
