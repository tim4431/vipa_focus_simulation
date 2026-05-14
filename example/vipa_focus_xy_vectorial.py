"""Vectorial XY focal plane profile for a VIPA field and high-NA objective."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from src.core import freespace_propagation, rays2elec2d
from src.vector_focus import focus_pupil_vectorial, polarization_observables
from src.vipa_focus import PARAMS_10_TWZ, vipa_rays

NA = 0.65
N_MEDIUM = 1.0
POLARIZATION = "x"
SOURCE_RESOLUTION = 40e-6
SOURCE_MARGIN_WAISTS = 8.0


def compact_source_extent(rays, margin_waists: float = SOURCE_MARGIN_WAISTS) -> float:
    """Choose the smallest square source window that contains the beam array."""
    max_x = max(abs(ray["x"]) + margin_waists * ray["w"] for ray in rays)
    max_y = max(abs(ray["y"]) + margin_waists * ray["w"] for ray in rays)
    return 2 * max(max_x, max_y)


def simulate_vipa_vector_xy(
    params: dict,
    NA: float = NA,
    n: float = N_MEDIUM,
    polarization=POLARIZATION,
    source_resolution: float = SOURCE_RESOLUTION,
    source_extent: float | None = None,
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
    params["RESOLUTION_X"] = source_resolution
    rays = vipa_rays(params)
    if source_extent is None:
        source_extent = compact_source_extent(rays)
    params["D"] = source_extent

    N_grid = int(params["D"] / (2 * params["RESOLUTION_X"])) * 2 + 1
    xi = np.linspace(-params["D"] / 2, params["D"] / 2, N_grid)
    yi = np.linspace(-params["D"] / 2, params["D"] / 2, N_grid)
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
    )

    extent_f = params["extent_f"]
    xf_mask = np.abs(xf) < extent_f
    yf_mask = np.abs(yf) < extent_f
    xf = xf[xf_mask]
    yf = yf[yf_mask]
    E_focus = E_focus[:, yf_mask, :][:, :, xf_mask]

    return xf, yf, E_focus, polarization_observables(E_focus), params


def plot_vipa_vector_xy(xf, yf, obs, NA: float = NA, zf: float = 0.0):
    extent_um = [xf[0] * 1e6, xf[-1] * 1e6, yf[0] * 1e6, yf[-1] * 1e6]
    images = [
        (r"$|E|^2$", obs["intensity"], "rainbow"),
        (r"$|E_x|^2$", obs["Ix"], "inferno"),
        (r"$|E_y|^2$", obs["Iy"], "inferno"),
        (r"$|E_z|^2$", obs["Iz"], "inferno"),
    ]

    vmax = obs["intensity"].max()
    vmin = max(vmax * 1e-6, np.finfo(float).tiny)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for ax, (title, image, cmap) in zip(axes.flat, images):
        im = ax.imshow(
            np.maximum(image, vmin),
            extent=extent_um,
            origin="lower",
            cmap=cmap,
            norm=(
                LogNorm(vmin=vmin, vmax=vmax)
                if image.max() > 0
                else Normalize(vmin=0, vmax=1)
            ),
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel(r"$x_f$ (um)")
        ax.set_ylabel(r"$y_f$ (um)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(rf"VIPA vector focus, NA={NA:.2f}, zf={zf * 1e6:.1f} um")
    plt.show()


if __name__ == "__main__":
    params = PARAMS_10_TWZ
    zf = 0.0

    xf, yf, E_focus, obs, used_params = simulate_vipa_vector_xy(
        params,
        NA=NA,
        n=N_MEDIUM,
        polarization=POLARIZATION,
        source_resolution=SOURCE_RESOLUTION,
        zf=zf,
    )

    FSR_L = params["lambda"] * params["f"] / params["d"]
    signal = obs["intensity"] > obs["intensity"].max() * 1e-4
    print("FSR_L =", FSR_L)
    print(f"source grid D = {used_params['D'] * 1e3:.2f} mm")
    print(f"source grid dx = {used_params['RESOLUTION_X'] * 1e6:.1f} um")
    print(f"focal pixel size = {(xf[1] - xf[0]) * 1e6:.2f} um")
    print(f"focus array shape = {E_focus.shape}")
    print(
        "peak longitudinal fraction in signal = "
        f"{obs['longitudinal_fraction'][signal].max():.3e}"
    )

    plot_vipa_vector_xy(xf, yf, obs, NA=NA, zf=zf)
