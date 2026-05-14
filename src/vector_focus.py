import numpy as np

from .core import calc_field_after_lens


def _validate_s_max(NA: float, n: float) -> float:
    s_max = NA / n
    if not (0 < s_max <= 1):
        raise ValueError("NA/n must be in (0, 1]")
    return s_max


def _as_grid_field(value, Xi: np.ndarray, Yi: np.ndarray, name: str) -> np.ndarray:
    if callable(value):
        value = value(Xi, Yi)

    arr = np.asarray(value, dtype=np.complex128)
    if arr.shape == ():
        return np.full(Xi.shape, arr, dtype=np.complex128)
    if arr.shape == Xi.shape:
        return arr

    try:
        return np.broadcast_to(arr, Xi.shape).astype(np.complex128, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a scalar, callable, or array broadcastable to {Xi.shape}"
        ) from exc


def _named_jones(name: str, Xi: np.ndarray, Yi: np.ndarray) -> np.ndarray:
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key in {"x", "horizontal", "linear_x"}:
        return np.stack([np.ones_like(Xi), np.zeros_like(Xi)]).astype(np.complex128)
    if key in {"y", "vertical", "linear_y"}:
        return np.stack([np.zeros_like(Xi), np.ones_like(Xi)]).astype(np.complex128)
    if key in {"right", "right_circular", "rhc", "sigma_minus"}:
        return np.stack([np.ones_like(Xi), -1j * np.ones_like(Xi)]) / np.sqrt(2)
    if key in {"left", "left_circular", "lhc", "sigma_plus"}:
        return np.stack([np.ones_like(Xi), 1j * np.ones_like(Xi)]) / np.sqrt(2)

    phi = np.arctan2(Yi, Xi)
    if key in {"radial", "linear_radial"}:
        return np.stack([np.cos(phi), np.sin(phi)]).astype(np.complex128)
    if key in {"azimuthal", "linear_azimuthal"}:
        return np.stack([-np.sin(phi), np.cos(phi)]).astype(np.complex128)

    raise ValueError(f"unknown named polarization {name!r}")


def _jones_on_grid(
    polarization,
    Xi: np.ndarray,
    Yi: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    if callable(polarization):
        polarization = polarization(Xi, Yi)

    if isinstance(polarization, str):
        pol = _named_jones(polarization, Xi, Yi)
    elif isinstance(polarization, (tuple, list)) and len(polarization) == 2:
        px = _as_grid_field(polarization[0], Xi, Yi, "polarization[0]")
        py = _as_grid_field(polarization[1], Xi, Yi, "polarization[1]")
        pol = np.stack([px, py])
    else:
        pol = np.asarray(polarization, dtype=np.complex128)
        if pol.shape == (2,):
            pol = np.stack(
                [
                    np.full(Xi.shape, pol[0], dtype=np.complex128),
                    np.full(Xi.shape, pol[1], dtype=np.complex128),
                ]
            )
        elif pol.shape == (2,) + Xi.shape:
            pass
        elif pol.shape == Xi.shape + (2,):
            pol = np.moveaxis(pol, -1, 0)
        else:
            raise ValueError(
                "polarization must be a 2-element Jones vector, a named "
                "polarization, a callable, or an array with shape "
                f"(2, {Xi.shape[0]}, {Xi.shape[1]})"
            )

    if normalize:
        norm = np.sqrt(np.sum(np.abs(pol) ** 2, axis=0))
        if np.all(norm == 0):
            raise ValueError("polarization cannot be zero everywhere")
        pol = pol / np.where(norm > 0, norm, 1.0)

    return pol


def make_pupil_grid(
    wavelength: float,
    NA: float,
    f: float,
    n: float = 1.0,
    N: int = 768,
    pupil_padding: float = 1.2,
    dx_focus: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the sine-condition pupil grid used by the Fourier lens transform.

    The returned `xi, yi` arrays are ordinary source-plane coordinates and are
    directly compatible with `core.calc_field_after_lens`. The clear aperture
    radius is `f * NA / n`; padding controls how much zero area surrounds it.
    """
    s_max = _validate_s_max(NA, n)
    if N < 2:
        raise ValueError("N must be at least 2")
    if pupil_padding <= 0:
        raise ValueError("pupil_padding must be positive")

    pupil_radius = f * s_max
    wavelength_medium = wavelength / n
    D = 2 * pupil_padding * pupil_radius
    if dx_focus is not None:
        if dx_focus <= 0:
            raise ValueError("dx_focus must be positive")
        D = max(D, wavelength_medium * f / dx_focus)

    xi = np.linspace(-D / 2, D / 2, N)
    yi = np.linspace(-D / 2, D / 2, N)
    return xi, yi


def pupil_geometry(
    xi: np.ndarray,
    yi: np.ndarray,
    NA: float,
    f: float,
    n: float = 1.0,
) -> dict:
    """
    Return high-NA pupil coordinates for an aplanatic sine-condition lens.

    `sx = xi / f` and `sy = yi / f` are direction cosines in the image-space
    medium. Points with `sqrt(sx**2 + sy**2) > NA/n` are outside the aperture.
    """
    s_max = _validate_s_max(NA, n)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    sx = Xi / f
    sy = Yi / f
    sin_theta = np.hypot(sx, sy)
    aperture = sin_theta <= s_max
    cos_theta = np.sqrt(np.clip(1 - sin_theta**2, 0, None))
    phi = np.arctan2(Yi, Xi)

    return {
        "Xi": Xi,
        "Yi": Yi,
        "rho": np.hypot(Xi, Yi),
        "sx": sx,
        "sy": sy,
        "sin_theta": sin_theta,
        "cos_theta": cos_theta,
        "phi": phi,
        "aperture": aperture,
        "s_max": s_max,
        "pupil_radius": f * s_max,
    }


def gaussian_pupil(
    waist: float,
    center: tuple[float, float] = (0.0, 0.0),
    phase=None,
):
    """
    Return a callable scalar pupil amplitude for a Gaussian beam.

    `phase` may be a scalar, array, or callable `phase(Xi, Yi)` in radians.
    """
    if waist <= 0:
        raise ValueError("waist must be positive")

    def _pupil(Xi: np.ndarray, Yi: np.ndarray) -> np.ndarray:
        Xc = Xi - center[0]
        Yc = Yi - center[1]
        field = np.exp(-(Xc**2 + Yc**2) / waist**2).astype(np.complex128)
        if phase is not None:
            phase_grid = _as_grid_field(phase, Xi, Yi, "phase")
            field *= np.exp(1j * phase_grid)
        return field

    return _pupil


def vectorial_pupil_field(
    xi: np.ndarray,
    yi: np.ndarray,
    scalar_pupil=1.0,
    polarization=(1.0, 0.0),
    NA: float = 1.0,
    f: float = 1.0,
    n: float = 1.0,
    aplanatic: bool = True,
    normalize_polarization: bool = True,
    return_geometry: bool = False,
):
    """
    Convert an arbitrary scalar/Jones pupil into Ex, Ey, Ez pupil components.

    Parameters
    ----------
    xi, yi
        1-D pupil/source-plane axes, in metres.
    scalar_pupil
        Scalar complex pupil amplitude. Accepts a scalar, an array on the
        `meshgrid(xi, yi)` grid, or a callable `scalar_pupil(Xi, Yi)`.
    polarization
        Input Jones field before the objective. Accepts a constant Jones vector,
        a pair of grid arrays/callables, a callable returning a Jones field,
        or one of: "x", "y", "left_circular", "right_circular", "radial",
        "azimuthal".
    aplanatic
        If true, applies the sine-condition Debye/Richards-Wolf apodization
        appropriate for using a Cartesian FFT over pupil coordinates.

    Returns
    -------
    E_pupil
        Complex array with shape `(3, len(yi), len(xi))`, component order
        `Ex, Ey, Ez`.
    """
    geom = pupil_geometry(xi, yi, NA=NA, f=f, n=n)
    Xi, Yi = geom["Xi"], geom["Yi"]
    aperture = geom["aperture"]
    cos_theta = geom["cos_theta"]
    sin_theta = geom["sin_theta"]
    phi = geom["phi"]
    cphi, sphi = np.cos(phi), np.sin(phi)

    scalar = _as_grid_field(scalar_pupil, Xi, Yi, "scalar_pupil")
    pol = _jones_on_grid(polarization, Xi, Yi, normalize=normalize_polarization)
    px, py = pol

    p_radial = px * cphi + py * sphi
    p_azimuthal = -px * sphi + py * cphi

    apodization = np.zeros_like(cos_theta, dtype=np.float64)
    if aplanatic:
        apodization[aperture] = 1 / np.sqrt(
            np.maximum(cos_theta[aperture], 1e-12)
        )
    else:
        apodization[aperture] = 1.0

    amp = scalar * apodization
    E_pupil = np.empty((3,) + Xi.shape, dtype=np.complex128)
    E_pupil[0] = amp * (p_radial * cos_theta * cphi - p_azimuthal * sphi)
    E_pupil[1] = amp * (p_radial * cos_theta * sphi + p_azimuthal * cphi)
    E_pupil[2] = amp * (-p_radial * sin_theta)

    if return_geometry:
        return E_pupil, geom
    return E_pupil


def calc_vector_field_after_lens(
    xi: np.ndarray,
    yi: np.ndarray,
    E_pupil: np.ndarray,
    wl: float,
    f: float,
    zf: float = 0.0,
    **kwargs,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Apply `core.calc_field_after_lens` independently to vector components.

    `E_pupil` must have shape `(n_components, len(yi), len(xi))`; for the
    vectorial high-NA helpers here, `n_components == 3`.
    """
    E_pupil = np.asarray(E_pupil, dtype=np.complex128)
    if E_pupil.ndim != 3 or E_pupil.shape[1:] != (len(yi), len(xi)):
        raise ValueError(
            "E_pupil must have shape (n_components, len(yi), len(xi))"
        )

    focused = []
    coords = None
    for component in E_pupil:
        coords, Ef = calc_field_after_lens(
            xi, yi, component.copy(), wl, f, zf=zf, **kwargs
        )
        focused.append(Ef)
    return coords, np.stack(focused)


def focus_pupil_vectorial(
    wavelength: float,
    NA: float,
    f: float,
    n: float = 1.0,
    scalar_pupil=1.0,
    polarization=(1.0, 0.0),
    xi: np.ndarray | None = None,
    yi: np.ndarray | None = None,
    N: int = 768,
    pupil_padding: float = 1.2,
    dx_focus: float | None = None,
    zf: float = 0.0,
    aplanatic: bool = True,
    normalize_polarization: bool = True,
    return_geometry: bool = False,
):
    """
    Focus an arbitrary polarized pupil field with the high-NA vectorial model.

    This is the general entry point. Provide `scalar_pupil` and `polarization`
    as arrays/callables to study non-Gaussian beams, spatially varying Jones
    fields, phase masks, or VIPA-like pupil fields. If `xi, yi` are omitted,
    a grid compatible with `core.calc_field_after_lens` is generated.

    Returns `(xf, yf), E_focus, E_pupil` by default. `E_focus` and `E_pupil`
    have shape `(3, len(y), len(x))` with component order `Ex, Ey, Ez`.
    """
    if xi is None or yi is None:
        if xi is not None or yi is not None:
            raise ValueError("xi and yi must be supplied together")
        xi, yi = make_pupil_grid(
            wavelength=wavelength,
            NA=NA,
            f=f,
            n=n,
            N=N,
            pupil_padding=pupil_padding,
            dx_focus=dx_focus,
        )

    E_pupil, geom = vectorial_pupil_field(
        xi,
        yi,
        scalar_pupil=scalar_pupil,
        polarization=polarization,
        NA=NA,
        f=f,
        n=n,
        aplanatic=aplanatic,
        normalize_polarization=normalize_polarization,
        return_geometry=True,
    )

    coords, E_focus = calc_vector_field_after_lens(
        xi, yi, E_pupil, wavelength / n, f, zf=zf
    )

    if return_geometry:
        return coords, E_focus, E_pupil, geom
    return coords, E_focus, E_pupil


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

    This convenience wrapper is implemented through `focus_pupil_vectorial`.
    Returns `(xf, yf), E_focus, E_pupil` where `E_*` has shape
    `(3, len(y), len(x))` and component order `Ex, Ey, Ez`. `wavelength` is
    the vacuum wavelength.
    """
    s_max = _validate_s_max(NA, n)
    pupil_radius = f * s_max
    if w_pupil is None:
        w_pupil = pupil_radius

    scalar_pupil = gaussian_pupil(w_pupil)
    return focus_pupil_vectorial(
        wavelength=wavelength,
        NA=NA,
        f=f,
        n=n,
        scalar_pupil=scalar_pupil,
        polarization=polarization,
        N=N,
        pupil_padding=pupil_padding,
        dx_focus=dx_focus,
        zf=zf,
        aplanatic=aplanatic,
    )


def polarization_observables(E: np.ndarray) -> dict:
    """
    Return intensity, component powers, longitudinal fraction, and Stokes maps.

    The Stokes parameters describe the transverse `(Ex, Ey)` projection. The
    total intensity and longitudinal fraction include `Ez`.
    """
    E = np.asarray(E, dtype=np.complex128)
    if E.ndim < 3 or E.shape[0] < 2:
        raise ValueError("E must have shape (2 or 3, ...)")

    Ex, Ey = E[0], E[1]
    Ez = E[2] if E.shape[0] > 2 else np.zeros_like(Ex)
    Ix, Iy, Iz = np.abs(Ex) ** 2, np.abs(Ey) ** 2, np.abs(Ez) ** 2
    transverse_intensity = Ix + Iy
    intensity = transverse_intensity + Iz
    eps = np.finfo(float).eps

    S0 = transverse_intensity
    S1 = Ix - Iy
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = 2 * np.imag(Ex * np.conj(Ey))
    Sden = np.maximum(S0, eps)

    return {
        "intensity": intensity,
        "transverse_intensity": transverse_intensity,
        "Ix": Ix,
        "Iy": Iy,
        "Iz": Iz,
        "longitudinal_fraction": Iz / np.maximum(intensity, eps),
        "S0": S0,
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "s1": S1 / Sden,
        "s2": S2 / Sden,
        "s3": S3 / Sden,
    }
