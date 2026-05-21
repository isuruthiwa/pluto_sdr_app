import numpy as np
import scipy.signal
from fractions import Fraction

FFT_SIZE = 2048
AUDIO_RATE = 48000
AFSK_MARK_HZ = 1200.0
AFSK_SPACE_HZ = 2200.0
AFSK_BAUD = 50.0
AFSK_MAX_PAYLOAD = 1024


def compute_fft(samples: np.ndarray, sample_rate: float,
                fft_size: int = FFT_SIZE, n_avg: int = 4) -> tuple:
    """Returns (freqs_relative_hz, power_db). freqs are centered at 0."""
    n = len(samples)
    n_ffts = max(1, n // fft_size)
    window = np.hanning(fft_size)
    win_power = np.sum(window ** 2)
    power = np.zeros(fft_size, dtype=np.float64)

    for i in range(min(n_avg, n_ffts)):
        chunk = samples[i * fft_size: (i + 1) * fft_size]
        if len(chunk) < fft_size:
            chunk = np.pad(chunk, (0, fft_size - len(chunk)))
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(chunk * window, fft_size))) ** 2
        power += spectrum

    power /= min(n_avg, n_ffts)
    power /= (win_power * fft_size)
    power_db = 10.0 * np.log10(np.maximum(power, 1e-14))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / sample_rate))
    return freqs, power_db.astype(np.float32)


def _tone(freq: float, num_samples: int, sample_rate: float) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _afsk_samples(bits: list[int], sample_rate: float) -> np.ndarray:
    samples_per_bit = int(round(sample_rate / AFSK_BAUD))
    mark = _tone(AFSK_MARK_HZ, samples_per_bit, sample_rate)
    space = _tone(AFSK_SPACE_HZ, samples_per_bit, sample_rate)
    audio = np.empty(samples_per_bit * len(bits), dtype=np.float32)
    for idx, bit in enumerate(bits):
        start = idx * samples_per_bit
        audio[start:start + samples_per_bit] = mark if bit else space
    return audio * 0.8


def encode_afsk_bytes(payload: bytes, sample_rate: float = AUDIO_RATE) -> np.ndarray:
    if len(payload) > AFSK_MAX_PAYLOAD:
        raise ValueError(f"Payload too large (max {AFSK_MAX_PAYLOAD} bytes)")
    framed = len(payload).to_bytes(2, 'little') + payload
    bits: list[int] = [1] * 16
    for byte in framed:
        bits.append(0)
        for bit_index in range(8):
            bits.append((byte >> bit_index) & 1)
        bits.append(1)
    return _afsk_samples(bits, sample_rate)


def _goertzel(samples: np.ndarray, sample_rate: float, freq: float) -> float:
    n = len(samples)
    k = int(0.5 + (n * freq) / sample_rate)
    omega = (2.0 * np.pi * k) / n
    cosine = np.cos(omega)
    coeff = 2.0 * cosine
    q0 = 0.0
    q1 = 0.0
    q2 = 0.0
    for sample in samples:
        q0 = coeff * q1 - q2 + sample
        q2 = q1
        q1 = q0
    return q1 * q1 + q2 * q2 - coeff * q1 * q2


def afsk_audio_to_fm(audio: np.ndarray, sample_rate: float = AUDIO_RATE,
                     deviation: float = 2500.0) -> np.ndarray:
    audio = audio.astype(np.float64)
    phase = 2.0 * np.pi * deviation * np.cumsum(audio) / sample_rate
    return np.exp(1j * phase).astype(np.complex64)


class AFSKDecoder:
    def __init__(self, sample_rate: float = AUDIO_RATE, baud: float = AFSK_BAUD):
        self.sample_rate = sample_rate
        self.baud = baud
        self.samples_per_bit = int(round(self.sample_rate / self.baud))
        self._buffer = np.zeros(0, dtype=np.float32)
        self._state = 'search'
        self._current_byte = 0
        self._bit_index = 0
        self._length = None
        self._length_bytes = bytearray()
        self._received = bytearray()
        self._previous_bit = 1

    def _decode_bit(self, samples: np.ndarray) -> int:
        samples = samples * np.hanning(len(samples))
        mark_energy = _goertzel(samples, self.sample_rate, AFSK_MARK_HZ)
        space_energy = _goertzel(samples, self.sample_rate, AFSK_SPACE_HZ)
        return 1 if mark_energy > space_energy else 0

    def feed(self, samples: np.ndarray) -> list[bytes]:
        self._buffer = np.concatenate((self._buffer, samples.astype(np.float32)))
        messages: list[bytes] = []

        while len(self._buffer) >= self.samples_per_bit:
            bit_window = self._buffer[:self.samples_per_bit]
            self._buffer = self._buffer[self.samples_per_bit:]
            bit = self._decode_bit(bit_window)

            if self._state == 'search':
                if self._previous_bit == 1 and bit == 0:
                    self._state = 'byte'
                    self._current_byte = 0
                    self._bit_index = 0
                self._previous_bit = bit
                continue

            if self._state == 'byte':
                if self._bit_index < 8:
                    self._current_byte |= (bit << self._bit_index)
                    self._bit_index += 1
                    continue

                if bit == 1:
                    if self._length is None:
                        self._length_bytes.append(self._current_byte)
                        if len(self._length_bytes) == 2:
                            self._length = int.from_bytes(self._length_bytes, 'little')
                            self._received = bytearray()
                            if self._length == 0:
                                messages.append(bytes(self._received))
                                self._length = None
                                self._length_bytes.clear()
                    else:
                        self._received.append(self._current_byte)
                        if len(self._received) >= self._length:
                            messages.append(bytes(self._received))
                            self._length = None
                            self._length_bytes.clear()
                self._state = 'search'
                self._previous_bit = bit

        return messages


def _lpf_taps(cutoff_hz: float, sample_rate: float, num_taps: int = 127) -> np.ndarray:
    nyq = sample_rate / 2.0
    cutoff_norm = min(cutoff_hz / nyq, 0.999)
    return scipy.signal.firwin(num_taps, cutoff_norm)


def _apply_lpf(signal: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    taps = _lpf_taps(cutoff_hz, sample_rate)
    return scipy.signal.lfilter(taps, [1.0], signal)


def _demod_fm(iq: np.ndarray, sample_rate: float, mode: str = 'nfm') -> np.ndarray:
    phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))
    if mode == 'wfm':
        deviation = 75e3
        audio = phase_diff / (2 * np.pi * deviation / sample_rate)
        # 75 µs de-emphasis
        tau = 75e-6
        alpha = 1.0 / (1.0 + tau * sample_rate)
        audio = scipy.signal.lfilter([alpha], [1.0, alpha - 1.0], audio)
    else:
        deviation = 5e3
        audio = phase_diff / (2 * np.pi * deviation / sample_rate)
    return audio.astype(np.float32)


def _demod_am(iq: np.ndarray) -> np.ndarray:
    audio = np.abs(iq).astype(np.float32)
    audio -= audio.mean()
    return audio


def _demod_ssb(iq: np.ndarray, upper: bool) -> np.ndarray:
    return np.real(iq if upper else np.conj(iq)).astype(np.float32)


def tune_and_demodulate(
    samples: np.ndarray,
    sample_rate: float,
    rx_offset: float,
    bandwidth: float,
    mode: str,
    audio_rate: int = AUDIO_RATE,
) -> np.ndarray:
    """Mix to rx_offset, filter, demodulate, resample to audio_rate."""
    n = len(samples)

    # Mix selected channel to DC
    if rx_offset != 0.0:
        t = np.arange(n, dtype=np.float64) / sample_rate
        samples = samples * np.exp(-2j * np.pi * rx_offset * t)

    # Channel bandwidth and intermediate sample rate
    if mode == 'wfm':
        ch_bw = min(bandwidth / 2.0, 100e3)
        inter_rate = 200e3
    elif mode in ('nfm', 'am'):
        ch_bw = min(bandwidth / 2.0, 12.5e3)
        inter_rate = 50e3
    else:  # usb / lsb
        ch_bw = min(bandwidth / 2.0, 3.5e3)
        inter_rate = 12e3

    filtered = _apply_lpf(samples, ch_bw, sample_rate)

    decim = max(1, int(sample_rate / inter_rate))
    decimated = filtered[::decim]
    actual_inter = sample_rate / decim

    if mode in ('nfm', 'wfm'):
        audio = _demod_fm(decimated, actual_inter, mode)
    elif mode == 'am':
        audio = _demod_am(decimated)
    elif mode == 'usb':
        audio = _demod_ssb(decimated, upper=True)
    elif mode == 'lsb':
        audio = _demod_ssb(decimated, upper=False)
    else:
        audio = np.real(decimated).astype(np.float32)

    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    # Resample to audio rate using rational approximation
    if int(actual_inter) != audio_rate:
        ratio = Fraction(audio_rate, int(actual_inter)).limit_denominator(200)
        audio = scipy.signal.resample_poly(audio, ratio.numerator, ratio.denominator)

    return audio.astype(np.float32)
