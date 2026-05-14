"""Photodetector response demo on a stored sequence (was sensor.py __main__)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.sensor import convolve_detector_response


if __name__ == "__main__":
    data = np.load(ROOT / "data/sequences_1.npz", allow_pickle=True)
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

    plt.figure()
    plt.imshow(
        intensity_profile,
        extent=[X.min() * 1e6, X.max() * 1e6, Y.min() * 1e6, Y.max() * 1e6],
        cmap="inferno",
        origin="lower",
    )
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

    t_out, V_out = convolve_detector_response(
        t, intensity_integrated, H, zero_pad=False
    )
    V_out = np.abs(V_out) ** 2
    Y = V_out[np.argmin(np.abs(t_out - 0.65e-6))]
    V_out /= Y
    print(len(t_out), len(t))
    np.savez(ROOT / "data/sequences_pd_output_1.npz", t=t_out, V=V_out)
    #
    plt.plot(t_out * 1e6, V_out)
    plt.xlabel("Time (us)")
    plt.ylabel("Normalized PD output voltage")
    plt.title("Simulated PD output voltage")
    plt.tight_layout()
    t_10 = t_out[np.where((V_out >= 0.1) & (t_out > 0.4e-6))[0][0]]
    t_90 = t_out[np.where((V_out >= 0.9) & (t_out > 0.4e-6))[0][0]]
    rise_time = t_90 - t_10
    print(f"10%-90% rise time (PD output): {rise_time*1e9:.2f} ns")
    plt.axvline(t_10 * 1e6, color="r", linestyle="--", label="10%")
    plt.axvline(t_90 * 1e6, color="g", linestyle="--", label="90%")
    plt.show()
