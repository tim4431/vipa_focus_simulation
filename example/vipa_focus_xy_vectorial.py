"""Vectorial XY focal plane profile for a VIPA field and high-NA objective."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.core import freespace_propagation, rays2elec2d
from src.vector_focus import focus_pupil_vectorial, polarization_observables
from src.vipa_focus import PARAMS_10_TWZ, vipa_rays

NA = 0.65
N_MEDIUM = 1.0
POLARIZATION = "x"


def simulate_vipa_vector_xy(
    params: dict,
    NA: float = NA,
    n: float = N_MEDIUM,
    polarization=POLARIZATION,
    zf: float = 0.0,
    tqdm_enable: bool = True,
):
    """
    Build the scalar VIPA pupil field, then focus it as a vector field.

    The scalar VIPA assembly is the same `rays2elec2d` path used by
    `vipa_focus_xy.py`; the high-NA objective projection is supplied by
    `focus_pupil_vectorial`.
    """
    params = params.copy()
    rays = vipa_rays(params)

    D = params["D"]
    RESOLUTION_X = params["RESOLUTION_X"]
    N_grid = int(D / (2 * RESOLUTION_X)) * 2 + 1
    xi = np.linspace(-D / 2, D / 2, N_grid)
    yi = np.linspace(-D / 2, D / 2, N_grid)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")

    Ei = rays2elec2d(Xi, Yi, rays, params, tqdm_enable=tqdm_enable)
    zfi = params.get("zfi", None)
    if zfi is not None:
        Ei = freespace_propagation(xi, yi, Ei, params["lambda"], zfi)

    (xf, yf), E_focus, _ = focus_pupil_vectorial(
        wavelength=params["lambda"],
        NA=NA,
        n=n,
        f=params["f"],
        xi=xi,
        yi=yi,
        scalar_pupil=Ei,
        polarization=polarization,
        zf=zf,
    )

    extent_f = params["extent_f"]
    xf_mask = np.abs(xf) < extent_f
    yf_mask = np.abs(yf) < extent_f
    xf = xf[xf_mask]
    yf = yf[yf_mask]
    E_focus = E_focus[:, yf_mask, :][:, :, xf_mask]

    return xf, yf, E_focus, polarization_observables(E_focus), params


def center_total_intensity(obs) -> float:
    cy = obs["intensity"].shape[0] // 2
    cx = obs["intensity"].shape[1] // 2
    center_intensity = float(np.real(obs["intensity"][cy, cx]))
    if center_intensity <= 0:
        raise ValueError("center total intensity must be positive for normalization")
    return center_intensity


def plot_vipa_vector_xy(xf, yf, obs, NA: float = NA, zf: float = 0.0):
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
    for col, (title, image) in enumerate(panels):
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
        ax.set_title(title)
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
    fig.suptitle(
        rf"VIPA vector focus, NA={NA:.2f}, zf={zf * 1e6:.1f} um"
    )
    plt.show()


if __name__ == "__main__":
    params = PARAMS_10_TWZ
    zf = 0.0

    xf, yf, E_focus, obs, used_params = simulate_vipa_vector_xy(
        params,
        NA=NA,
        n=N_MEDIUM,
        polarization=POLARIZATION,
        zf=zf,
    )

    FSR_L = params["lambda"] * params["f"] / params["d"]
    signal = obs["intensity"] > obs["intensity"].max() * 1e-4
    print("FSR_L =", FSR_L)
    print(f"source grid D = {used_params['D'] * 1e3:.2f} mm")
    print(f"source grid dx = {used_params['RESOLUTION_X'] * 1e6:.1f} um")
    print(
        "source grid shape = "
        f"{int(used_params['D'] / (2 * used_params['RESOLUTION_X'])) * 2 + 1}"
    )
    print(f"focal pixel size = {(xf[1] - xf[0]) * 1e6:.2f} um")
    print(f"focus array shape = {E_focus.shape}")
    print(f"center total intensity normalization = {center_total_intensity(obs):.3e}")
    print(
        "peak longitudinal fraction in signal = "
        f"{obs['longitudinal_fraction'][signal].max():.3e}"
    )

    plot_vipa_vector_xy(xf, yf, obs, NA=NA, zf=zf)
