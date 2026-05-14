"""Focal-plane polarization of a Gaussian beam focused by a high-NA objective."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.vector_focus import (
    focus_pupil_vectorial,
    gaussian_pupil,
    polarization_observables,
)


def center_total_intensity(obs) -> float:
    cy = obs["intensity"].shape[0] // 2
    cx = obs["intensity"].shape[1] // 2
    center_intensity = float(np.real(obs["intensity"][cy, cx]))
    if center_intensity <= 0:
        raise ValueError("center total intensity must be positive for normalization")
    return center_intensity


def plot_component_rows(xf, yf, obs, title: str):
    extent_um = [xf[0] * 1e6, xf[-1] * 1e6, yf[0] * 1e6, yf[-1] * 1e6]
    norm_intensity = center_total_intensity(obs)
    panels = [
        (r"$|E_x|^2$", obs["Ix"] / norm_intensity),
        (r"$|E_y|^2$", obs["Iy"] / norm_intensity),
        (r"$|E_z|^2$", obs["Iz"] / norm_intensity),
    ]

    component_vmax = max(image.max() for _, image in panels)
    component_vmin = max(component_vmax * 1e-6, np.finfo(float).tiny)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), constrained_layout=True)

    linear_im = None
    log_im = None
    for col, (panel_title, image) in enumerate(panels):
        ax = axes[0, col]
        linear_im = ax.imshow(
            image,
            extent=extent_um,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=component_vmax,
            interpolation="nearest",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")

        ax = axes[1, col]
        log_im = ax.imshow(
            np.maximum(image, component_vmin),
            extent=extent_um,
            origin="lower",
            cmap="inferno",
            norm=LogNorm(vmin=component_vmin, vmax=component_vmax),
            interpolation="nearest",
        )
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")

    axes[0, 0].text(0.02, 0.95, "linear", transform=axes[0, 0].transAxes, color="w")
    axes[1, 0].text(0.02, 0.95, "log", transform=axes[1, 0].transAxes, color="w")
    label = "component intensity / center total intensity"
    fig.colorbar(linear_im, ax=axes[0, :], fraction=0.046, label=label)
    fig.colorbar(log_im, ax=axes[1, :], fraction=0.046, label=label)
    fig.suptitle(title)
    plt.show()


if __name__ == "__main__":
    wavelength = 780e-9
    NA = 0.65
    n = 1.0
    f = 3e-3
    pupil_radius = f * NA / n
    pupil = gaussian_pupil(0.85 * pupil_radius)

    # `focus_pupil_vectorial` is the general interface: replace `pupil` with
    # any scalar pupil array/callable and `polarization` with any Jones map.
    (xf, yf), E, _ = focus_pupil_vectorial(
        wavelength=wavelength,
        NA=NA,
        n=n,
        f=f,
        scalar_pupil=pupil,
        polarization="x",
        N=1024,
        dx_focus=50e-9,
    )
    mx = np.abs(xf) <= 3e-6
    my = np.abs(yf) <= 3e-6
    xf, yf, E = xf[mx], yf[my], E[:, my][:, :, mx]

    obs = polarization_observables(E)
    signal = obs["intensity"] > obs["intensity"].max() * 1e-4

    print(f"focal pixel size: {(xf[1] - xf[0]) * 1e9:.1f} nm")
    print(
        "peak longitudinal fraction: "
        f"{np.max(obs['longitudinal_fraction'][signal]):.3f}"
    )
    print(
        "on-axis longitudinal fraction: "
        f"{obs['longitudinal_fraction'][len(yf)//2, len(xf)//2]:.3e}"
    )
    print(f"center total intensity normalization: {center_total_intensity(obs):.3e}")

    plot_component_rows(
        xf,
        yf,
        obs,
        rf"Gaussian vector focus, NA={NA:.2f}",
    )
