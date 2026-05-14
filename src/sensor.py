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


if __name__ == "__main__":
    data = np.load("./data/sequences_1.npz", allow_pickle=True)
    # integrate intensity within 75umx75um square
    extent_x = 75e-6 / 2
    center_x = -1.4e-6
    center_y = 7.4e-6
    X, Y = data["X"], data["Y"]
    Z, t = data["Z"], data["t"]
    mask = (np.abs(X - center_x) <= extent_x) & (np.abs(Y - center_y) <= extent_x)
    intensity_integrated = np.sum(Z[:, mask], axis=1)
    intensity_integrated /= np.max(intensity_integrated)
    # catch t closest to 0.503us and show the internsity profile
    idx = np.argmin(np.abs(t - 0.573e-6))
    intensity_profile = Z[idx, :, :]
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(
        intensity_profile,
        extent=[X.min() * 1e6, X.max() * 1e6, Y.min() * 1e6, Y.max() * 1e6],
        cmap="inferno",
        origin="lower",
    )
    # and also show mask boundary
    # add rectangle
    from matplotlib.patches import Rectangle

    plt.gca().add_patch(
        Rectangle(
            ((center_x - extent_x) * 1e6, (center_y - extent_x) * 1e6),
            2 * extent_x * 1e6,
            2 * extent_x * 1e6,
            linewidth=1,
            fill=False,
            edgecolor="cyan",
        )
    )
    plt.colorbar(label="Intensity (arb. units)")
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.title(f"Intensity profile at t={t[idx]*1e6:.3f} us")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(t * 1e6, intensity_integrated)
    plt.xlabel("Time (us)")
    plt.ylabel("Normalized integrated intensity")
    plt.title("Integrated intensity within 75um x 75um square")
    # get 10%-90% rise time
    t_10 = t[np.where(intensity_integrated >= 0.1)[0][0]]
    t_90 = t[np.where(intensity_integrated >= 0.9)[0][0]]
    rise_time = t_90 - t_10
    print(f"10%-90% rise time: {rise_time*1e9:.2f} ns")
    plt.axvline(t_10 * 1e6, color="r", linestyle="--", label="10%")
    plt.axvline(t_90 * 1e6, color="g", linestyle="--", label="90%")

    def H(f):
        f0 = 50e6
        return 1 / (1 + (1j * f / f0) ** 3)

    # from sensor import convolve_detector_response

    t_out, V_out = convolve_detector_response(
        t, intensity_integrated, H, zero_pad=False
    )
    V_out = np.abs(V_out) ** 2
    Y = V_out[np.argmin(np.abs(t_out - 0.65e-6))]
    V_out /= Y
    print(len(t_out), len(t))
    np.savez("./data/sequences_pd_output_1.npz", t=t_out, V=V_out)
    #
    plt.plot(t_out * 1e6, V_out)
    plt.xlabel("Time (us)")
    plt.ylabel("Normalized PD output voltage")
    plt.title("Simulated PD output voltage")
    plt.tight_layout()
    # calculate 10%-90% rise time from 0.1Y to 0.9Y
    t_10 = t_out[np.where((V_out >= 0.1) & (t_out > 0.4e-6))[0][0]]
    t_90 = t_out[np.where((V_out >= 0.9) & (t_out > 0.4e-6))[0][0]]
    rise_time = t_90 - t_10
    print(f"10%-90% rise time (PD output): {rise_time*1e9:.2f} ns")
    plt.axvline(t_10 * 1e6, color="r", linestyle="--", label="10%")
    plt.axvline(t_90 * 1e6, color="g", linestyle="--", label="90%")
    plt.show()
