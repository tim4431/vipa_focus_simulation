"""True-3D VIPA xz simulation.

Builds the xz image by running a full 2-D `crosssection_xy_partial` at z = 0
to locate the focal-spot y-position y0, then extracts the row at y = y0 from
a partial xy DFT at every z. This is the memory-efficient way to get a true
3-D xz slice (as opposed to `crosssection_xz`, which is a y_f≈0 line-cut).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from src.vipa_focus import *
from src.crosssections import crosssection_xy_partial


def xz_via_xy(rays, params, z_scan, xf_targets, yf_targets):
    """Build a (len(xf_targets), len(z_scan)) xz image via partial xy DFTs."""
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


if __name__ == "__main__":
    params = PARAMS_80_TWZ

    EXTENT_Z = 10e-6
    NZ = 100
    M_X = 401
    M_Y = 201

    extent_f = params["extent_f"]
    xf_targets = np.linspace(-extent_f, extent_f, M_X)
    yf_targets = np.linspace(-extent_f, extent_f, M_Y)

    rays = vipa_rays(params)
    z_scan = np.linspace(-EXTENT_Z, EXTENT_Z, NZ)
    print("Running partial xy DFT at each z (memory-efficient)...")
    xf, yf, y0, profiles = xz_via_xy(rays, params, z_scan, xf_targets, yf_targets)
    print(f"profiles shape: {profiles.shape}")

    out = ROOT / "data/vipa_focus_xz.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, xf=xf, yf=yf, y0=y0, z_scan=z_scan, profiles=profiles, extent_f=extent_f
    )
    print(f"✓  Saved {out}")
