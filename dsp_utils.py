import numpy as np
import scipy.signal
from fractions import Fraction

FFT_SIZE = 2048
AUDIO_RATE = 48000


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
