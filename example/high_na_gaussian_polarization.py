"""Focal-plane polarization of a Gaussian beam focused by a high-NA objective."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.vector_focus import focus_gaussian_vectorial, polarization_observables


if __name__ == "__main__":
    wavelength = 780e-9
    NA = 0.8
    n = 1.0
    f = 3e-3
    pupil_radius = f * NA / n

    (xf, yf), E, _ = focus_gaussian_vectorial(
        wavelength=wavelength,
        NA=NA,
        n=n,
        f=f,
        w_pupil=0.85 * pupil_radius,
        polarization=(1, 0),
        N=2048,
        dx_focus=50e-9,
    )
    mx = np.abs(xf) <= 3e-6
    my = np.abs(yf) <= 3e-6
    xf, yf, E = xf[mx], yf[my], E[:, my][:, :, mx]

    obs = polarization_observables(E)
    signal = obs["intensity"] > obs["intensity"].max() * 1e-4

    print(f"focal pixel size: {(xf[1] - xf[0]) * 1e9:.1f} nm")
    print(f"peak longitudinal fraction: {np.max(obs['longitudinal_fraction'][signal]):.3f}")
    print(f"on-axis longitudinal fraction: {obs['longitudinal_fraction'][len(yf)//2, len(xf)//2]:.3e}")

    panels = [
        (r"$|E_x|^2$", obs["Ix"]),
        (r"$|E_y|^2$", obs["Iy"]),
        (r"$|E_z|^2$", obs["Iz"]),
    ]
    component_vmax = max(obs["Ix"].max(), obs["Iy"].max(), obs["Iz"].max())
    component_vmin = component_vmax * 1e-6
    extent_um = [xf[0] * 1e6, xf[-1] * 1e6, yf[0] * 1e6, yf[-1] * 1e6]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    component_im = None
    for ax, (title, image) in zip(axes, panels):
        im = ax.imshow(
            np.maximum(image, component_vmin),
            extent=extent_um,
            origin="lower",
            cmap="inferno",
            norm=LogNorm(vmin=component_vmin, vmax=component_vmax),
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        component_im = im
    fig.colorbar(component_im, ax=axes, fraction=0.046, label="component intensity")

    plt.show()
