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


if __name__ == "__main__":

    params = PARAMS_10
    # params = PARAMS_80_TWZ
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
