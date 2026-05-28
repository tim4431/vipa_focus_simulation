"""True-3D VIPA xz comparison vs. an ideal Gaussian beam.

Unlike `ripa_vs_gaussian_focus.py`, which builds the xz image from
`crosssection_xz` (a y_f≈0 line-cut for each z), this script runs a full 2-D
`crosssection_xy_partial` at every z and extracts the row at y = y0, where y0
is the y-position of the focal spot found by stacking |E|^2 along x at z = 0.

CLI:
    python example/ripa_vs_gaussian_focus_3d.py            # calc + plot
    python example/ripa_vs_gaussian_focus_3d.py --mode calc   # compute & save npz, no plots
    python example/ripa_vs_gaussian_focus_3d.py --mode plot   # load npz & plot only

Three output PNGs (dpi=600):
- ripa_vs_gaussian_focus_3d.png       : stacked Gaussian / VIPA xz heatmaps
- ripa_vs_gaussian_focus_3d_xcut.png  : 1-D x line-cut at z = 0 (Gaussian fits)
- ripa_vs_gaussian_focus_3d_zcut.png  : 1-D z line-cut at x = 0 (Rayleigh check)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.vipa_focus import *
from src.crosssections import crosssection_xy_partial
from src.fit_gaussian import fit_gaussian_1d


DATA_PATH = ROOT / "data/ripa_vs_gaussian_focus_3d.npz"


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


def xz_via_xy(rays, params, z_scan, xf_targets, yf_targets):
    """Build a (len(xf_targets), len(z_scan)) xz image via partial xy DFTs.

    Step 1 (z=0): full (xf, yf) intensity map → stack along x to find y0.
    Step 2 (z≠0): only the y=y0 row is needed, so call the partial DFT with
                  yf_targets=[y0]. This is much cheaper than the full xy map.
    """
    xf, yf, _, intensity_xy0 = crosssection_xy_partial(
        rays, params, xf_targets, yf_targets, zf=0.0
    )
    x_stacked = np.sum(intensity_xy0, axis=1)
    iy0 = int(np.argmax(x_stacked))
    y0 = yf[iy0]
    print(f"  located y0 = {y0*1e6:.3f} µm (iy0 = {iy0} / {len(yf)})")

    profiles = np.empty((len(xf), len(z_scan)), dtype=float)
    iz0 = int(np.argmin(np.abs(z_scan - 0.0)))
    profiles[:, iz0] = intensity_xy0[iy0, :]
    del intensity_xy0

    yf_row = np.array([y0])
    for iz, z in enumerate(tqdm(z_scan, desc="xy at each z")):
        if iz == iz0:
            continue
        _, _, _, intensity_row = crosssection_xy_partial(
            rays, params, xf_targets, yf_row, zf=float(z)
        )
        profiles[:, iz] = intensity_row[0, :]

    return xf, yf, y0, profiles


def run_calc(params, extent_z, nz, m_x, m_y):
    extent_f = params["extent_f"]
    xf_targets = np.linspace(-extent_f, extent_f, m_x)
    yf_targets = np.linspace(-extent_f, extent_f, m_y)

    rays = vipa_rays(params)
    z_scan = np.linspace(-extent_z, extent_z, nz)
    print("Running partial xy DFT at each z (memory-efficient)...")
    xf, yf, y0, profiles = xz_via_xy(rays, params, z_scan, xf_targets, yf_targets)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_PATH,
        xf=xf,
        yf=yf,
        y0=y0,
        z_scan=z_scan,
        profiles=profiles,
        extent_f=extent_f,
    )
    print(f"✓  Saved {DATA_PATH}")
    return xf, yf, y0, z_scan, profiles, extent_f


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run with --mode calc (or default) first."
        )
    d = np.load(DATA_PATH)
    xf = d["xf"]
    yf = d["yf"]
    y0 = float(d["y0"])
    z_scan = d["z_scan"]
    profiles = d["profiles"]
    extent_f = float(d["extent_f"]) if "extent_f" in d.files else float(np.max(np.abs(xf)))
    print(f"✓  Loaded {DATA_PATH}")
    return xf, yf, y0, z_scan, profiles, extent_f


def run_plot(xf, z_scan, profiles, extent_f, w0, wl, x0, z0, cmap):
    vipa_img = profiles / np.max(profiles)
    gauss_img = gaussian_beam_xz(xf, z_scan, w0, wl, x0=x0, z0=z0)
    gauss_img = gauss_img / np.max(gauss_img)

    extent_z = float(np.max(np.abs(z_scan)))
    extent = [
        (-extent_z - z0) * 1e6,
        (extent_z - z0) * 1e6,
        (-extent_f - x0) * 1e6,
        (extent_f - x0) * 1e6,
    ]

    # -------- panel 1: stacked heatmaps ----------------------------------
    fig, (ax_g, ax_v) = plt.subplots(
        2, 1, figsize=(3.2, 3.8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, img in ((ax_g, gauss_img), (ax_v, vipa_img)):
        im = ax.imshow(
            img,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax.set_ylabel(r"$x$ (µm)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax_v.set_xlabel(r"$z$ (µm)")

    fig.colorbar(
        im,
        ax=[ax_g, ax_v],
        ticks=[0, 1],
        orientation="horizontal",
        location="bottom",
        pad=0.02,
        fraction=0.05,
        shrink=0.8,
    )

    out_png = ROOT / "example/render/ripa_vs_gaussian_focus_3d.png"
    fig.savefig(out_png, dpi=600, transparent=True)
    print(f"✓  Saved {out_png}")

    # -------- panel 2: 1-D x line-cut at z = z0 --------------------------
    iz0 = int(np.argmin(np.abs(z_scan - z0)))
    vipa_cut = vipa_img[:, iz0]
    gauss_cut = gauss_img[:, iz0]

    print("\nFitting VIPA x-cut...")
    w_vipa, _ = fit_waist(xf, vipa_cut)
    print("Fitting Gaussian-reference x-cut...")
    w_gauss, _ = fit_waist(xf, gauss_cut)
    print(
        f"\nWaist (1/e^2 intensity radius):\n"
        f"  VIPA      : {w_vipa*1e9:.1f} nm\n"
        f"  Gaussian  : {w_gauss*1e9:.1f} nm   (expected {w0*1e9:.0f} nm)"
    )

    fig2, ax2 = plt.subplots(figsize=(3.2, 2.2))
    ax2.plot(xf * 1e6, gauss_cut, color="tab:blue", lw=2,
             label=rf"Gaussian, $w_0$={w_gauss*1e9:.0f} nm")
    ax2.plot(xf * 1e6, vipa_cut, color="tab:red", lw=2, ls="--",
             label=rf"VIPA, $w_0$={w_vipa*1e9:.0f} nm")
    ax2.set_xlabel(r"$x$ (µm)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.set_xlim(-extent_f * 1e6, extent_f * 1e6)
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.legend(frameon=False, loc="upper right")
    fig2.tight_layout()

    out_png2 = ROOT / "example/render/ripa_vs_gaussian_focus_3d_xcut.png"
    fig2.savefig(out_png2, dpi=600, transparent=True)
    print(f"✓  Saved {out_png2}")

    # -------- panel 3: 1-D z line-cut at x = x0 --------------------------
    ix0 = int(np.argmin(np.abs(xf - x0)))
    vipa_zcut = vipa_img[ix0, :]
    gauss_zcut = gauss_img[ix0, :]

    print("\nFitting VIPA z-cut...")
    zw_vipa, _ = fit_waist(z_scan, vipa_zcut)
    print("Fitting Gaussian-reference z-cut...")
    zw_gauss, _ = fit_waist(z_scan, gauss_zcut)
    zR_expected = np.pi * w0**2 / wl
    print(
        f"\nAxial 1/e^2 half-width (from Gaussian fit):\n"
        f"  VIPA      : {zw_vipa*1e6:.2f} µm\n"
        f"  Gaussian  : {zw_gauss*1e6:.2f} µm\n"
        f"  (Rayleigh range zR = pi w0^2/lambda = {zR_expected*1e6:.2f} µm)"
    )

    fig3, ax3 = plt.subplots(figsize=(3.2, 2.2))
    ax3.plot(z_scan * 1e6, gauss_zcut, color="tab:blue", lw=2, label="Gaussian")
    ax3.plot(z_scan * 1e6, vipa_zcut, color="tab:red", lw=2, ls="--", label="VIPA")
    ax3.axvline(zR_expected * 1e6, color="0.6", lw=0.8, ls=":")
    ax3.axvline(-zR_expected * 1e6, color="0.6", lw=0.8, ls=":",
                label=rf"$\pm z_R$ = {zR_expected*1e6:.2f} µm")
    ax3.set_xlabel(r"$z$ (µm)")
    ax3.set_ylabel("Intensity (a.u.)")
    ax3.set_xlim(-extent_z * 1e6, extent_z * 1e6)
    ax3.set_ylim(0, 1.05)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.legend(frameon=False, loc="upper right")
    fig3.tight_layout()

    out_png3 = ROOT / "example/render/ripa_vs_gaussian_focus_3d_zcut.png"
    fig3.savefig(out_png3, dpi=600, transparent=True)
    print(f"✓  Saved {out_png3}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["calc", "plot", "both"],
        default="both",
        help="calc: only simulate and save npz. plot: only load npz and plot. both: do both (default).",
    )
    args = parser.parse_args()

    params = PARAMS_80_TWZ
    EXTENT_Z = 10e-6
    NZ = 100
    M_X = 401
    M_Y = 201

    X0 = 0.0
    Z0 = 0.0
    CMAP = "Blues"

    W0 = 720e-9
    WL = 780e-9

    if args.mode in ("calc", "both"):
        xf, yf, y0, z_scan, profiles, extent_f = run_calc(
            params, EXTENT_Z, NZ, M_X, M_Y
        )
        if args.mode == "calc":
            print("✓  calc-only mode; skipping plots.")
            sys.exit(0)
    else:  # plot
        xf, yf, y0, z_scan, profiles, extent_f = load_data()

    run_plot(xf, z_scan, profiles, extent_f, W0, WL, X0, Z0, CMAP)
