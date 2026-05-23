"""True-3D VIPA xz comparison vs. an ideal Gaussian beam.

Unlike `ripa_vs_gaussian_focus.py`, which builds the xz image from
`crosssection_xz` (a y_f≈0 line-cut for each z), this script runs a full 2-D
`crosssection_xy` at every z and extracts the row at y = y0, where y0 is the
y-position of the focal spot found by stacking |E|^2 along x at z = 0.

Three output PNGs (dpi=600):
- ripa_vs_gaussian_focus_3d.png       : stacked Gaussian / VIPA xz heatmaps
- ripa_vs_gaussian_focus_3d_xcut.png  : 1-D x line-cut at z = 0 (Gaussian fits)
- ripa_vs_gaussian_focus_3d_zcut.png  : 1-D z line-cut at x = 0 (Rayleigh check)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.vipa_focus import *
from src.crosssections import crosssection_xy
from src.fit_gaussian import fit_gaussian_1d


def gaussian_beam_xz(xf, z_scan, w0, wl, x0=0.0, z0=0.0):
    """|E|^2 of a TEM00 Gaussian beam in the x–z plane.

    Returns a 2-D array shaped (len(xf), len(z_scan)) so it matches our
    `profiles` convention (rows = x_f, cols = z_f).
    """
    zR = np.pi * w0**2 / wl
    z = z_scan - z0
    w = w0 * np.sqrt(1.0 + (z / zR) ** 2)
    x = (xf - x0)[:, None]
    w = w[None, :]
    return (w0 / w) ** 2 * np.exp(-2.0 * x**2 / w**2)


def fit_waist(x, intensity):
    """Fit |E|^2(x) with a Gaussian + offset; return (waist 1/e^2, popt)."""
    popt = fit_gaussian_1d(x, intensity, offset=True)
    sigma = abs(popt[1])
    return 2.0 * sigma, popt


def xz_via_xy(rays, params, z_scan):
    """Build a (len(xf), len(z_scan)) xz image by full xy simulations.

    Locates y0 at z=0 by stacking |E|^2 along x and taking the max, then
    extracts the y=y0 row from each per-z xy intensity map.
    """
    # locate y0 at z=0
    xf, yf, _, intensity_xy0 = crosssection_xy(rays, params, zf=0.0, show_focus=False)
    x_stacked = np.sum(intensity_xy0, axis=1)  # (len(yf),) — sum over x
    iy0 = int(np.argmax(x_stacked))
    y0 = yf[iy0]
    print(f"  located y0 = {y0*1e6:.3f} µm (iy0 = {iy0} / {len(yf)})")

    profiles = np.empty((len(xf), len(z_scan)), dtype=float)
    profiles[:, int(np.argmin(np.abs(z_scan - 0.0)))] = intensity_xy0[iy0, :]

    for iz, z in enumerate(tqdm(z_scan, desc="xy at each z")):
        if z == 0.0:
            continue  # already done
        _, _, _, intensity_xy = crosssection_xy(rays, params, zf=z, show_focus=False)
        profiles[:, iz] = intensity_xy[iy0, :]

    return xf, yf, y0, profiles


if __name__ == "__main__":
    params = PARAMS_80_TWZ

    EXTENT_Z = 10e-6
    NZ = 100  # full xy at each z is expensive — keep modest by default

    # plot centered at this (x_f, z_f) point (metres)
    X0 = 0.0
    Z0 = 0.0
    CMAP = "Blues"

    # ideal Gaussian-beam reference
    W0 = 720e-9
    WL = 780e-9

    rays = vipa_rays(params)
    z_scan = np.linspace(-EXTENT_Z, EXTENT_Z, NZ)
    print("Running full xy cross-section at each z (this is slow)...")
    xf, yf, y0, profiles = xz_via_xy(rays, params, z_scan)

    vipa_img = profiles / np.max(profiles)
    gauss_img = gaussian_beam_xz(xf, z_scan, W0, WL, x0=X0, z0=Z0)
    gauss_img = gauss_img / np.max(gauss_img)

    np.savez(
        ROOT / "data/ripa_vs_gaussian_focus_3d.npz",
        xf=xf,
        yf=yf,
        y0=y0,
        z_scan=z_scan,
        profiles=profiles,
    )

    extent_f = params["extent_f"]
    extent = [
        (-EXTENT_Z - Z0) * 1e6,
        (EXTENT_Z - Z0) * 1e6,
        (-extent_f - X0) * 1e6,
        (extent_f - X0) * 1e6,
    ]

    # -------- panel 1: stacked heatmaps ----------------------------------
    fig, (ax_g, ax_v) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, sharey=True)
    for ax, img in ((ax_g, gauss_img), (ax_v, vipa_img)):
        im = ax.imshow(
            img,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=CMAP,
            vmin=0,
            vmax=1,
        )
        ax.set_ylabel(r"$x$ (µm)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax_v.set_xlabel(r"$z$ (µm)")

    cbar = fig.colorbar(
        im,
        ax=[ax_g, ax_v],
        ticks=[0, 1],
        orientation="horizontal",
        location="bottom",
        pad=0.02,
        fraction=0.05,
        shrink=0.8,
    )
    cbar.set_label("Intensity (a.u.)")

    out_png = ROOT / "example/render/ripa_vs_gaussian_focus_3d.png"
    fig.savefig(out_png, dpi=600)
    print(f"✓  Saved {out_png}")

    # -------- panel 2: 1-D x line-cut at z = 0 ---------------------------
    iz0 = int(np.argmin(np.abs(z_scan - Z0)))
    vipa_cut = vipa_img[:, iz0]
    gauss_cut = gauss_img[:, iz0]

    print("\nFitting VIPA x-cut...")
    w_vipa, _ = fit_waist(xf, vipa_cut)
    print("Fitting Gaussian-reference x-cut...")
    w_gauss, _ = fit_waist(xf, gauss_cut)
    print(
        f"\nWaist (1/e^2 intensity radius):\n"
        f"  VIPA      : {w_vipa*1e9:.1f} nm\n"
        f"  Gaussian  : {w_gauss*1e9:.1f} nm   (expected {W0*1e9:.0f} nm)"
    )

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(
        xf * 1e6,
        gauss_cut,
        color="tab:blue",
        lw=2,
        label=rf"Gaussian, $w_0$={w_gauss*1e9:.0f} nm",
    )
    ax2.plot(
        xf * 1e6,
        vipa_cut,
        color="tab:red",
        lw=2,
        ls="--",
        label=rf"VIPA, $w_0$={w_vipa*1e9:.0f} nm",
    )
    ax2.set_xlabel(r"$x$ (µm)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.set_xlim(-extent_f * 1e6, extent_f * 1e6)
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.legend(frameon=False, loc="upper right")
    fig2.tight_layout()

    out_png2 = ROOT / "example/render/ripa_vs_gaussian_focus_3d_xcut.png"
    fig2.savefig(out_png2, dpi=600)
    print(f"✓  Saved {out_png2}")

    # -------- panel 3: 1-D z line-cut at x = 0 ---------------------------
    ix0 = int(np.argmin(np.abs(xf - X0)))
    vipa_zcut = vipa_img[ix0, :]
    gauss_zcut = gauss_img[ix0, :]

    print("\nFitting VIPA z-cut...")
    zw_vipa, _ = fit_waist(z_scan, vipa_zcut)
    print("Fitting Gaussian-reference z-cut...")
    zw_gauss, _ = fit_waist(z_scan, gauss_zcut)
    zR_expected = np.pi * W0**2 / WL
    print(
        f"\nAxial 1/e^2 half-width (from Gaussian fit):\n"
        f"  VIPA      : {zw_vipa*1e6:.2f} µm\n"
        f"  Gaussian  : {zw_gauss*1e6:.2f} µm\n"
        f"  (Rayleigh range zR = pi w0^2/lambda = {zR_expected*1e6:.2f} µm)"
    )

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(
        z_scan * 1e6,
        gauss_zcut,
        color="tab:blue",
        lw=2,
        label="Gaussian",
    )
    ax3.plot(
        z_scan * 1e6,
        vipa_zcut,
        color="tab:red",
        lw=2,
        ls="--",
        label="VIPA",
    )
    ax3.axvline(zR_expected * 1e6, color="0.6", lw=0.8, ls=":")
    ax3.axvline(
        -zR_expected * 1e6,
        color="0.6",
        lw=0.8,
        ls=":",
        label=rf"$\pm z_R$ = {zR_expected*1e6:.2f} µm",
    )
    ax3.set_xlabel(r"$z$ (µm)")
    ax3.set_ylabel("Intensity (a.u.)")
    ax3.set_xlim(-EXTENT_Z * 1e6, EXTENT_Z * 1e6)
    ax3.set_ylim(0, 1.05)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.legend(frameon=False, loc="upper right")
    fig3.tight_layout()

    out_png3 = ROOT / "example/render/ripa_vs_gaussian_focus_3d_zcut.png"
    fig3.savefig(out_png3, dpi=600)
    print(f"✓  Saved {out_png3}")

    plt.show()
