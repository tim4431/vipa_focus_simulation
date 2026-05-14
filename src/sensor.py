import numpy as np


def convolve_detector_response(
    t: np.ndarray,
    signal: np.ndarray,
    H_func,
    *,
    deconvolve: bool = False,
    regularization: float = 1e-10,
    zero_pad: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply (convolve) or remove (deconvolve) a detector frequency response
    to/from a time-domain signal.

    Parameters
    ----------
    t : ndarray, shape (N,)
        Time samples [s], must be uniformly spaced.
    signal : ndarray, shape (N,)
        Input signal amplitude vs time.
        - If deconvolve=False: Optical Intensity/Power (Input to PD).
        - If deconvolve=True:  PD Voltage (Output from PD).
    H_func : callable
        Function H_func(f) -> complex ndarray.
        Returns the complex frequency response H(f) of the system.
    deconvolve : bool, optional
        If False (default), simulates the output voltage (Convolution): V = I * H.
        If True, reconstructs the input intensity (Deconvolution): I = V / H.
    regularization : float, optional
        Used only if deconvolve=True. Adds a small constant to the denominator
        to prevent division by zero or noise amplification at frequencies where
        H(f) is small.
        Formula: H_inv = conj(H) / (|H|^2 + regularization).
    zero_pad : bool, optional
        If True, zero-pad to the next power of two for speed and to reduce
        circular convolution artifacts.

    Returns
    -------
    t_out : ndarray, shape (M,)
        Time axis for the output.
    output_signal : ndarray, shape (M,)
        - If deconvolve=False: Simulated Voltage [V].
        - If deconvolve=True:  Reconstructed Intensity [W].
    """

    t = np.asarray(t).ravel()
    signal = np.asarray(signal).ravel()

    if t.shape != signal.shape:
        raise ValueError("t and signal must have the same shape")

    # Check uniform sampling
    dt_array = np.diff(t)
    dt = dt_array.mean()
    if not np.allclose(dt_array, dt, rtol=1e-4, atol=1e-12):
        raise ValueError("Time array t must be uniformly sampled")

    N = t.size

    # Optional zero-padding
    if zero_pad:
        M = 1 << int(np.ceil(np.log2(2 * N)))
    else:
        M = N

    # Build padded signal array
    sig_pad = np.zeros(M, dtype=float)
    sig_pad[:N] = signal

    # New time axis
    t_out = np.arange(M) * dt + t[0]

    # Frequency grid
    freqs = np.fft.fftfreq(M, d=dt)

    # Evaluate Transfer function H(f)
    H = H_func(freqs)
    if H.shape != freqs.shape:
        raise ValueError("H_func(freqs) must return array of same shape as freqs")

    # FFT of the input
    sig_f = np.fft.fft(sig_pad)

    # Apply Transfer Function
    if not deconvolve:
        # Convolution: V(f) = I(f) * H(f)
        out_f = sig_f * H
    else:
        # Deconvolution: I(f) = V(f) / H(f)
        # Using Tikhonov-style regularization to handle H(f) ≈ 0
        # Formula: Signal / H  ~=  Signal * conj(H) / (|H|^2 + reg)
        denom = (np.abs(H) ** 2) + regularization
        out_f = sig_f * np.conj(H) / denom

    # Back to time domain
    output_signal = np.fft.ifft(out_f)
    output_signal = np.real_if_close(output_signal)

    return t_out, output_signal

