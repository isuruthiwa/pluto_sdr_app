import queue
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

AUDIO_RATE = 48000
_QUEUE_MAX = 16  # max buffered chunks before dropping


class MicInputWorker(QThread):
    """Captures audio from microphone and emits chunks for TX processing."""

    audio_chunk = pyqtSignal(np.ndarray)  # float32, 48kHz audio
    error_msg = pyqtSignal(str)
    status_msg = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._stream = None
        self._chunk_size = 4096  # ~85ms at 48kHz
        self._level_db = -100.0  # signal level in dB

    @property
    def level_db(self):
        """Current microphone level in dB (RMS)."""
        return self._level_db

    def set_chunk_size(self, size: int):
        """Set capture buffer size in samples."""
        self._chunk_size = max(256, min(16384, size))

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        if not _SD_OK:
            self.error_msg.emit("sounddevice not installed — mic input disabled")
            return

        self._running = True
        try:
            with sd.InputStream(
                samplerate=AUDIO_RATE,
                channels=1,
                dtype='float32',
                blocksize=self._chunk_size,
            ) as stream:
                self._stream = stream
                self.status_msg.emit("Microphone input started")

                while self._running:
                    # Blocking read of one chunk
                    chunk, overflowed = stream.read(self._chunk_size)
                    
                    if overflowed:
                        self.error_msg.emit("Microphone buffer overflow")
                    
                    # Flatten to 1D and ensure float32
                    chunk = chunk.reshape(-1).astype(np.float32)
                    
                    # Calculate RMS level in dB
                    rms = np.sqrt(np.mean(chunk ** 2))
                    self._level_db = 20.0 * np.log10(rms + 1e-12)
                    
                    # Emit audio chunk for TX processing
                    self.audio_chunk.emit(chunk)

        except Exception as exc:
            self.error_msg.emit(f"Microphone error: {exc}")
        finally:
            self._stream = None
            self.status_msg.emit("Microphone input stopped")
