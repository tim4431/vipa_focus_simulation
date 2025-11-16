import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from functools import lru_cache
import time


def calc_field_after_lens(
    xi: np.ndarray,  # 1-D focal-plane x-axis
    yi: np.ndarray,  # 1-D focal-plane y-axis
    Ei: np.ndarray,  # 2-D input-plane field
    wl: float,  # wavelength (metres)
    f: float,  # focal length (metres)
    zf: float = 0.0,  # focal plane offset
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the complex field (or intensity) at an *arbitrary* focal-plane
    grid using a single FFT.

    The supplied xf / yf *must* be uniformly sampled and monotonic,
    but they need not be square or share the same resolution.

    Returns
    -------
    xf,yf   : 1-D arrays (metres)
    field   : 2-D array  shape (len(yf), len(xf))

    -----------
    input plane(i),E              lens              output plane(f),E_tilde   output plane(zf)
    (xi,yi)                       (kxi,kyi)         (kxf,kyf)                 (xf,yf)
    we have kxi = 2*pi*xf/(lambda*f) and kxf = -2*pi*xi/(lambda*f)

    propagate it to zr using E
    a.  get xi,yi
    b.  calculate Ei(xi,yi) and kxi,kyi
    c.  get kxf = -2*pi*xi/(lambda*f) (remember the negative sign)
    d.  get xf = lambda*f*kxi/(2*pi)
    e.  get Ei(xi,yi) = Ei(z=0,kxf,kyf)
    f.  propagate it to zf using the Ei(z=zf,kxf,kyf) = Ei(z=0,kxf,kyf)*exp(j*kzf*zf)
    g.  fourier transform Ei_tilde(z=zf,xf,yf) = Fourier(Ei_tilde(z=zf,kxf,kyf))
    """

    k0 = 2 * np.pi / wl

    # a
    # Xi, Yi = np.meshgrid(xi, yi, indexing="xy")  # make sure the indexing is xy
    # b
    # Ei = elec2d(Xi, Yi, params, alpha, mmax)
    show_field = kwargs.get("show_field", False)
    if show_field:
        plt.figure(figsize=(6, 5))
        plt.imshow(
            np.abs(Ei) ** 2,
            extent=[
                np.min(xi) * 1e6,
                np.max(xi) * 1e6,
                np.min(yi) * 1e6,
                np.max(yi) * 1e6,
            ],
            origin="lower",
            cmap="rainbow",
            aspect="equal",
            vmin=0,
        )
        plt.colorbar(label="Intensity (arb.)")
        plt.xlabel(r"$x_i$ (input plane, $\mu$m)")
        plt.ylabel(r"$y_i$ (input plane, $\mu$m)")
        plt.title("Electric field in the input plane")
        plt.tight_layout()
        plt.show()

    # deal with 1d data
    if len(xi) > 1:
        dxi = xi[1] - xi[0]
        kxi = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(xi), d=dxi))
        xf = wl * f * kxi / (2 * np.pi)
    else:
        xf = 0
        kxi = 0
    if len(yi) > 1:
        dyi = yi[1] - yi[0]
        kyi = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(yi), d=dyi))
        yf = wl * f * kyi / (2 * np.pi)
    else:
        yf = 0
        kyi = 0

    # cd
    kxf = -2 * np.pi * xi / (wl * f)  # negative sign
    kyf = 2 * np.pi * yi / (wl * f)

    if np.abs(zf) > 0:
        # ef
        KXf, KYf = np.meshgrid(kxf, kyf)
        kzf = np.sqrt(k0**2 - KXf**2 - KYf**2 + 0j)
        Ei *= np.exp(1j * kzf * zf)
    # g
    E_tilde = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ei), norm="ortho"))

    return (xf, yf), E_tilde


@lru_cache(maxsize=128)
def _compute_kzf(
    xi_hash: int,
    yi_hash: int,
    len_xi: int,
    len_yi: int,
    dxi: float,
    dyi: float,
    wl: float,
):
    """
    Cached computation of kzf array for freespace propagation.

    Parameters
    ----------
    xi_hash : int
        Hash of xi array (from xi.tobytes())
    yi_hash : int
        Hash of yi array (from yi.tobytes())
    len_xi : int
        Length of xi array
    len_yi : int
        Length of yi array
    dxi : float
        Spacing in xi
    dyi : float
        Spacing in yi
    wl : float
        Wavelength (metres)

    Returns
    -------
    kzf : 2-D array
        Propagation wavevector in z-direction
    """
    k0 = 2 * np.pi / wl

    # Calculate spatial frequencies
    kxi = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len_xi, d=dxi))
    kyi = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len_yi, d=dyi))

    KX, KY = np.meshgrid(kxi, kyi)
    kzf = np.sqrt(k0**2 - KX**2 - KY**2 + 0j)

    return kzf


def freespace_propagation(
    xi: np.ndarray,
    yi: np.ndarray,
    Ei: np.ndarray,
    wl: float,
    z: float,
) -> np.ndarray:
    """
    Propagate a field Ei at (xi,yi) a distance z in free space using
    the angular spectrum method.

    This version caches the computation of kzf when xi, yi, and wl are the same,
    avoiding redundant calculations for repeated calls with the same grid.

    Parameters
    ----------
    xi : 1-D array
        x-coordinates of the input plane (metres)
    yi : 1-D array
        y-coordinates of the input plane (metres)
    Ei : 2-D array
        complex field at the input plane
    wl : float
        wavelength (metres)
    z : float
        propagation distance (metres)

    Returns
    -------
    Ef : 2-D array
        complex field at the output plane after propagation
    """
    # Calculate grid parameters
    if len(yi) > 1:
        dxi = xi[1] - xi[0]
        dyi = yi[1] - yi[0]

        # Create hashable identifiers for xi and yi arrays
        xi_hash = hash(xi.tobytes())
        yi_hash = hash(yi.tobytes())

        # Get cached kzf or compute it
        kzf = _compute_kzf(xi_hash, yi_hash, len(xi), len(yi), dxi, dyi, wl)
    else:
        dxi = xi[1] - xi[0]

        k0 = 2 * np.pi / wl
        # Calculate spatial frequencies
        kxi = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(xi), d=dxi))
        kyi = 0

        KX, KY = np.meshgrid(kxi, kyi)
        kzf = np.sqrt(k0**2 - KX**2 - KY**2 + 0j)

    # Fourier transform of the input field
    Ei_k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ei), norm="ortho"))

    # Apply propagation phase factor
    Ei_k *= np.exp(1j * kzf * z)

    # Inverse Fourier transform to get the propagated field
    Ef = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Ei_k), norm="ortho"))

    return Ef
