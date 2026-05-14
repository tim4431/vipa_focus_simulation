import numpy as np

from .core import calc_field_after_lens


def _unit_jones(polarization):
    pol = np.asarray(polarization, dtype=np.complex128)
    if pol.shape != (2,):
        raise ValueError("polarization must be a 2-element Jones vector")
    norm = np.linalg.norm(pol)
    if norm == 0:
        raise ValueError("polarization cannot be zero")
    return pol / norm


def focus_gaussian_vectorial(
    wavelength: float,
    NA: float,
    f: float,
    n: float = 1.0,
    w_pupil: float | None = None,
    polarization=(1.0, 0.0),
    N: int = 768,
    pupil_padding: float = 1.2,
    dx_focus: float | None = None,
    zf: float = 0.0,
    aplanatic: bool = True,
):
    """
    Vectorial Richards-Wolf focus of a Gaussian beam through a high-NA objective.

    Returns (xf, yf), E_focus, E_pupil where E_* has shape (3, len(y), len(x))
    and component order is Ex, Ey, Ez. `wavelength` is the vacuum wavelength.
    """
    s_max = NA / n
    if not (0 < s_max <= 1):
        raise ValueError("NA/n must be in (0, 1]")

    pupil_radius = f * s_max
    if w_pupil is None:
        w_pupil = pupil_radius

    wavelength_medium = wavelength / n
    D = 2 * pupil_padding * pupil_radius
    if dx_focus is not None:
        if dx_focus <= 0:
            raise ValueError("dx_focus must be positive")
        D = max(D, wavelength_medium * f / dx_focus)

    xi = np.linspace(-D / 2, D / 2, N)
    yi = np.linspace(-D / 2, D / 2, N)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")

    rho = np.hypot(Xi, Yi)
    sx = Xi / f
    sy = Yi / f
    sin_theta = np.hypot(sx, sy)
    aperture = sin_theta <= s_max
    cos_theta = np.sqrt(np.clip(1 - sin_theta**2, 0, None))
    phi = np.arctan2(Yi, Xi)
    cphi, sphi = np.cos(phi), np.sin(phi)

    px, py = _unit_jones(polarization)
    p_radial = px * cphi + py * sphi
    p_azimuthal = -px * sphi + py * cphi

    amp = np.exp(-(rho / w_pupil) ** 2) * aperture
    if aplanatic:
        amp = amp / np.sqrt(np.maximum(cos_theta, 1e-12))

    E_pupil = np.empty((3, N, N), dtype=np.complex128)
    E_pupil[0] = amp * (p_radial * cos_theta * cphi - p_azimuthal * sphi)
    E_pupil[1] = amp * (p_radial * cos_theta * sphi + p_azimuthal * cphi)
    E_pupil[2] = amp * (-p_radial * sin_theta)

    focused = []
    coords = None
    for i in range(3):
        coords, Ef = calc_field_after_lens(
            xi, yi, E_pupil[i].copy(), wavelength_medium, f, zf=zf
        )
        focused.append(Ef)
    return coords, np.stack(focused), E_pupil


def polarization_observables(E):
    """Return component intensities and longitudinal fraction."""
    Ex, Ey, Ez = E
    Ix, Iy, Iz = np.abs(Ex) ** 2, np.abs(Ey) ** 2, np.abs(Ez) ** 2
    intensity = Ix + Iy + Iz
    eps = np.finfo(float).eps
    return {
        "intensity": intensity,
        "Ix": Ix,
        "Iy": Iy,
        "Iz": Iz,
        "longitudinal_fraction": Iz / np.maximum(intensity, eps),
    }
