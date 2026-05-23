"""Compare Gaussian and VIPA high-NA vectorial focal-plane polarization."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.core import freespace_propagation, rays2elec2d
from src.vector_focus import (
    focus_pupil_vectorial,
    gaussian_pupil,
    polarization_observables,
)
from src.vipa_focus import PARAMS_10_TWZ, vipa_rays

FOV = 10e-6
WAVELENGTH = 780e-9
OBJECTIVE_NA = 0.65
NA = OBJECTIVE_NA  # Backwards-compatible alias for the objective aperture.
N_MEDIUM = 1.0
POLARIZATION = "x"
FOCAL_LENGTH = PARAMS_10_TWZ["f"]
GAUSSIAN_FOCUS_WAIST = 1.1e-6 * (FOCAL_LENGTH / 0.017)
GAUSSIAN_EFFECTIVE_NA = WAVELENGTH / (np.pi * GAUSSIAN_FOCUS_WAIST)
RENDER_DIR = ROOT / "example" / "render"


def gaussian_pupil_waist_for_focus(
    wavelength: float,
    f: float,
    waist_focus: float,
    n: float = N_MEDIUM,
) -> float:
    """Pupil 1/e field radius giving a Gaussian focal 1/e^2 waist."""
    if waist_focus <= 0:
        raise ValueError("waist_focus must be positive")
    return (wavelength / n) * f / (np.pi * waist_focus)


def crop_field_to_fov(
    xf: np.ndarray,
    yf: np.ndarray,
    E: np.ndarray,
    fov: float = FOV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_fov = fov / 2
    x_mask = np.abs(xf) <= half_fov
    y_mask = np.abs(yf) <= half_fov
    return xf[x_mask], yf[y_mask], E[:, y_mask, :][:, :, x_mask]


def center_total_intensity(xf: np.ndarray, yf: np.ndarray, obs: dict) -> float:
    cx = int(np.argmin(np.abs(xf)))
    cy = int(np.argmin(np.abs(yf)))
    center_intensity = float(np.real(obs["intensity"][cy, cx]))
    if center_intensity <= 0:
        raise ValueError("center total intensity must be positive for normalization")
    return center_intensity


def component_energy_fractions(obs: dict) -> dict:
    total_energy = float(np.sum(obs["intensity"]))
    if total_energy <= 0:
        raise ValueError("total intensity must be positive")
    return {
        "x": float(np.sum(obs["Ix"]) / total_energy),
        "y": float(np.sum(obs["Iy"]) / total_energy),
        "z": float(np.sum(obs["Iz"]) / total_energy),
    }


def component_energy_fractions_from_field(E: np.ndarray) -> dict:
    component_energies = [float(np.vdot(component, component).real) for component in E]
    total_energy = sum(component_energies)
    if total_energy <= 0:
        raise ValueError("total field energy must be positive")
    return {
        "x": component_energies[0] / total_energy,
        "y": component_energies[1] / total_energy,
        "z": component_energies[2] / total_energy,
    }


def format_percent(value: float) -> str:
    return f"{100 * value:.3f}%"


def format_fraction_line(label: str, fractions: dict) -> str:
    return (
        rf"{label}: "
        + rf"$E_y$={format_percent(fractions['y'])}, "
        + rf"$E_z$={format_percent(fractions['z'])}"
    )


def plot_component_rows(
    xf: np.ndarray,
    yf: np.ndarray,
    obs: dict,
    title: str,
    fov: float = FOV,
    save_path: Path | str | None = None,
    dpi: int = 600,
):
    half_fov_um = fov * 1e6 / 2
    extent_um = [xf[0] * 1e6, xf[-1] * 1e6, yf[0] * 1e6, yf[-1] * 1e6]
    norm_intensity = center_total_intensity(xf, yf, obs)
    fractions = component_energy_fractions(obs)
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

        ax = axes[1, col]
        log_im = ax.imshow(
            np.maximum(image, component_vmin),
            extent=extent_um,
            origin="lower",
            cmap="inferno",
            norm=LogNorm(vmin=component_vmin, vmax=component_vmax),
            interpolation="nearest",
        )

    for ax in axes.flat:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_xlim(-half_fov_um, half_fov_um)
        ax.set_ylim(-half_fov_um, half_fov_um)
        ax.set_xticks([-half_fov_um, 0, half_fov_um])
        ax.set_yticks([-half_fov_um, 0, half_fov_um])

    axes[0, 0].text(0.02, 0.95, "linear", transform=axes[0, 0].transAxes, color="w")
    axes[1, 0].text(0.02, 0.95, "log", transform=axes[1, 0].transAxes, color="w")
    label = "component intensity / center total intensity"
    fig.colorbar(linear_im, ax=axes[0, :], fraction=0.046, label=label)
    fig.colorbar(log_im, ax=axes[1, :], fraction=0.046, label=label)
    fraction_lines = format_fraction_line("FOV energy fractions", fractions)
    fig.suptitle(title + "\n" + fraction_lines)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {save_path}")
    plt.show()


def simulate_gaussian_vector_xy(
    wavelength: float = WAVELENGTH,
    objective_na: float = OBJECTIVE_NA,
    n: float = N_MEDIUM,
    f: float = FOCAL_LENGTH,
    waist_focus: float = GAUSSIAN_FOCUS_WAIST,
    fov: float = FOV,
    N: int = 1024,
    dx_focus: float = 50e-9,
    polarization=POLARIZATION,
    return_full_metrics: bool = False,
):
    pupil_waist = gaussian_pupil_waist_for_focus(wavelength, f, waist_focus, n=n)
    pupil = gaussian_pupil(pupil_waist)
    (xf, yf), E, _ = focus_pupil_vectorial(
        wavelength=wavelength,
        NA=objective_na,
        n=n,
        f=f,
        scalar_pupil=pupil,
        polarization=polarization,
        N=N,
        dx_focus=dx_focus,
    )
    full_metrics = {
        "energy_fractions": component_energy_fractions_from_field(E),
        "shape": E.shape,
        "xf": xf,
        "yf": yf,
    }
    xf, yf, E = crop_field_to_fov(xf, yf, E, fov=fov)
    result = (xf, yf, E, polarization_observables(E))
    if return_full_metrics:
        return result + (full_metrics,)
    return result


def simulate_vipa_vector_xy(
    params: dict,
    objective_na: float = OBJECTIVE_NA,
    n: float = N_MEDIUM,
    polarization=POLARIZATION,
    fov: float = FOV,
    zf: float = 0.0,
    tqdm_enable: bool = True,
    return_full_metrics: bool = False,
):
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
        NA=objective_na,
        n=n,
        f=params["f"],
        xi=xi,
        yi=yi,
        scalar_pupil=Ei,
        polarization=polarization,
        zf=zf,
    )
    full_metrics = {
        "energy_fractions": component_energy_fractions_from_field(E_focus),
        "shape": E_focus.shape,
        "xf": xf,
        "yf": yf,
    }
    xf, yf, E_focus = crop_field_to_fov(xf, yf, E_focus, fov=fov)
    result = (xf, yf, E_focus, polarization_observables(E_focus), params)
    if return_full_metrics:
        return result + (full_metrics,)
    return result


def print_summary(
    label: str,
    xf: np.ndarray,
    yf: np.ndarray,
    E: np.ndarray,
    obs: dict,
):
    fractions = component_energy_fractions(obs)
    print(label)
    print(f"  focal pixel size = {(xf[1] - xf[0]) * 1e9:.1f} nm")
    print(f"  focus array shape = {E.shape}")
    print(
        f"  center total intensity normalization = {center_total_intensity(xf, yf, obs):.3e}"
    )
    print(f"  FOV energy fraction y = {format_percent(fractions['y'])}")
    print(f"  FOV energy fraction z = {format_percent(fractions['z'])}")


if __name__ == "__main__":
    xg, yg, Eg, obs_g = simulate_gaussian_vector_xy()
    print_summary("Gaussian high-NA focus", xg, yg, Eg, obs_g)
    gaussian_pupil_waist = gaussian_pupil_waist_for_focus(
        WAVELENGTH, FOCAL_LENGTH, GAUSSIAN_FOCUS_WAIST, n=N_MEDIUM
    )
    print(f"  focal length = {FOCAL_LENGTH * 1e3:.1f} mm")
    print(f"  target waist = {GAUSSIAN_FOCUS_WAIST * 1e6:.2f} um")
    print(f"  Gaussian effective NA = {GAUSSIAN_EFFECTIVE_NA:.3f}")
    print(f"  Gaussian pupil waist = {gaussian_pupil_waist * 1e3:.3f} mm")
    plot_component_rows(
        xg,
        yg,
        obs_g,
        rf"Gaussian vector focus, $w_0$={1.18e-6 * 1e6:.2f} um",
        save_path=RENDER_DIR / "high_na_gaussian_polarization.png",
    )

    xv, yv, Ev, obs_v, used_params = simulate_vipa_vector_xy(PARAMS_10_TWZ)
    print_summary("VIPA high-NA focus", xv, yv, Ev, obs_v)
    print(f"  source grid D = {used_params['D'] * 1e3:.2f} mm")
    print(f"  source grid dx = {used_params['RESOLUTION_X'] * 1e6:.1f} um")
    plot_component_rows(
        xv,
        yv,
        obs_v,
        "RIPA vector focus, $\sqrt{w_x w_y}$=1.18 um",
        save_path=RENDER_DIR / "high_na_vipa_polarization.png",
    )
