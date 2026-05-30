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

CWD = Path.cwd()
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.vipa_focus import *
from src.crosssections import crosssection_xy_partial


def xz_via_xy(rays, params, z_scan, xf_targets, yf_targets):
    """Build a (len(xf_targets), len(z_scan)) xz image via partial xy DFTs.

    Step 1 (z=0): full (xf, yf) intensity map → stack along x to find y0.
    Step 2 (z≠0): only the y=y0 row is needed, so call the partial DFT with
                  yf_targets=[y0]. This is much cheaper than the full xy map.
    """
    xf, yf, field_xy0, intensity_xy0 = crosssection_xy_partial(
        rays, params, xf_targets, yf_targets, zf=0.0
    )
    x_stacked = np.sum(intensity_xy0, axis=1)
    iy0 = int(np.argmax(x_stacked))
    y0 = yf[iy0]
    print(f"  located y0 = {y0*1e6:.3f} µm (iy0 = {iy0} / {len(yf)})")

    profiles = np.empty((len(xf), len(z_scan)), dtype=float)
    profiles_field = np.empty((len(xf), len(z_scan)), dtype=complex)
    iz0 = int(np.argmin(np.abs(z_scan - 0.0)))
    profiles[:, iz0] = intensity_xy0[iy0, :]
    profiles_field[:, iz0] = field_xy0[iy0, :]

    del intensity_xy0
    del field_xy0

    yf_row = np.array([y0])
    for iz, z in enumerate(tqdm(z_scan, desc="xy at each z")):
        if iz == iz0:
            continue
        _, _, field_row, intensity_row = crosssection_xy_partial(
            rays, params, xf_targets, yf_row, zf=float(z)
        )
        profiles[:, iz] = intensity_row[0, :]
        profiles_field[:, iz] = field_row[0, :]

    return xf, yf, y0, profiles, profiles_field


def run_calc(params, extent_z, nz, m_x, m_y):
    extent_f = params["extent_f"]
    xf_targets = np.linspace(-extent_f, extent_f, m_x)
    yf_targets = np.linspace(-extent_f, extent_f, m_y)

    rays = vipa_rays(params)
    z_scan = np.linspace(-extent_z, extent_z, nz)
    print("Running partial xy DFT at each z (memory-efficient)...")
    xf, yf, y0, profiles, profiles_field = xz_via_xy(
        rays, params, z_scan, xf_targets, yf_targets
    )

    return xf, yf, y0, z_scan, profiles, profiles_field, extent_f


if __name__ == "__main__":

    params = PARAMS_10_TWZ2
    M = 1000
    # M = 20000
    params["lambda"] = params["lambda"] * M
    params["extent_f"] = params["extent_f"] * M * 5
    EXTENT_Z = 20e-6 * M
    print(EXTENT_Z / params["f"])
    print(EXTENT_Z / params["lambda"])
    from time import sleep

    sleep(2)
    NZ = 200
    M_X = 201
    M_Y = 201
    #
    # params = PARAMS_80_TWZ
    # params.update({"extent_f": 100e-6})
    # EXTENT_Z = 150e-6
    # NZ = 500
    # M_X = 501
    # M_Y = 501

    xf, yf, y0, z_scan, profiles, profiles_field, extent_f = run_calc(
        params, EXTENT_Z, NZ, M_X, M_Y
    )

    DATA_PATH = CWD / "render/ripa_params_10_xa.npz"
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_PATH,
        xf=xf,
        yf=yf,
        y0=y0,
        z_scan=z_scan,
        profiles=profiles,
        profiles_field=profiles_field,
        extent_f=extent_f,
    )
    print(f"✓  Saved {DATA_PATH}")
    # also plot countour arg(profiles_field)~0
    phase_field = np.mod(np.angle(profiles_field), 2 * np.pi)
    phase_close_to_zero = -np.abs(phase_field - np.pi)
    extent = (z_scan[0] * 1e6, z_scan[-1] * 1e6, -extent_f * 1e6, extent_f * 1e6)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(
        profiles,
        extent=extent,
        aspect="auto",
        cmap="inferno",
    )
    fig.colorbar(im0, ax=axes[0], label="intensity (a.u.)")
    axes[0].set_xlabel("z (µm)")
    axes[0].set_ylabel("x (µm)")
    axes[0].set_title("Intensity |E|²")

    im1 = axes[1].imshow(
        phase_field,
        # phase_close_to_zero,
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=2 * np.pi,
    )
    fig.colorbar(im1, ax=axes[1], label="phase (rad)")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("z (mm)")
    axes[1].set_title("Phase of the field (mod 2π)")

    plt.tight_layout()
    plt.savefig(CWD / "render/ripa_params_10_phase.png", dpi=600)
    print(f"✓  Saved {CWD / 'render/ripa_params_10_phase.png'}")
