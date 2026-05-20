import queue
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

AUDIO_RATE = 48000
_QUEUE_MAX  = 8     # max buffered chunks before dropping


class AudioWorker(QThread):
    """Drains an audio queue and writes to sounddevice output stream."""

    error_msg = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q       = queue.Queue(maxsize=_QUEUE_MAX)
        self._running = False
        self._volume  = 0.8
        self._squelch = -60.0    # dB threshold; signal below this → silence
        self._muted   = False
        self._stream  = None

    # ── public API ──────────────────────────────────────────────────────────

    def enqueue(self, audio: np.ndarray):
        """Called from SDR thread. Drops oldest chunk if queue is full."""
        if not self._running or self._muted or not _SD_OK:
            return
        try:
            self._q.put_nowait(audio)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(audio)
            except queue.Full:
                pass

    def set_volume(self, vol: float):
        """0.0 – 1.0"""
        self._volume = max(0.0, min(1.0, vol))

    def set_squelch(self, threshold_db: float):
        self._squelch = threshold_db

    def set_muted(self, muted: bool):
        self._muted = muted

    def stop(self):
        self._running = False
        self.wait(2000)

    # ── thread ──────────────────────────────────────────────────────────────

    def run(self):
        if not _SD_OK:
            self.error_msg.emit("sounddevice not installed — audio disabled")
            return

        self._running = True
        try:
            with sd.OutputStream(
                samplerate=AUDIO_RATE,
                channels=1,
                dtype='float32',
                blocksize=1024,
            ) as stream:
                self._stream = stream
                while self._running:
                    try:
                        chunk = self._q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if self._muted:
                        continue

                    # Squelch gate
                    rms_db = 20.0 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-12)
                    if rms_db < self._squelch:
                        chunk = np.zeros_like(chunk)

                    chunk = (chunk * self._volume).astype(np.float32)
                    # Clip to prevent distortion
                    np.clip(chunk, -1.0, 1.0, out=chunk)
                    stream.write(chunk.reshape(-1, 1))

        except Exception as exc:
            self.error_msg.emit(f"Audio error: {exc}")
        finally:
            self._stream = None
