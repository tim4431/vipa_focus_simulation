"""XZ profile vs. phi animation (was vipa_focus.py TYPE == 2)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import imageio

from src.vipa_focus import PARAMS_80, vipa_rays
from src.crosssections import crosssection_xz


if __name__ == "__main__":
    params = PARAMS_80

    EXTENT_Z = 20e-3
    NZ = 2000
    NPHI = 30
    gif_data = []
    for phi in np.linspace(0.0, 2 * np.pi, NPHI):
        print(f"phi = {phi:.2f}")
        params["phi"] = phi  # update phase

        rays = vipa_rays(params)
        z_scan, xf, profiles = crosssection_xz(
            rays, params, extent_z=EXTENT_Z, n_z=NZ, show_focus=False
        )
        gif_data.append(profiles)
    gif_data = np.array(gif_data)  # shape (NPHI, n_xf, n_z)
    # normalize to [0,255]
    gif_data = gif_data / np.max(gif_data)
    np.save(ROOT / "data/vipa_focus_xz_phi_scan.npy", gif_data)
    gif_data = (gif_data * 255).astype(np.uint8)
    # kron the gif to make xz aspect equal
    dx = xf[1] - xf[0]
    dz = z_scan[1] - z_scan[0]
    ratio = dz / dx
    print(f"dx={dx*1e6:.2f} um, dz={dz*1e6:.2f} um, ratio={ratio:.2f}")
    # gif_data = np.kron(gif_data, np.ones((1, 1, int(ratio)), dtype=np.uint8))
    # ----- write the animated GIF -----------------------------------------
    imageio.mimsave(ROOT / "example/render/scan_phi.gif", gif_data, fps=5, loop=0)
    print("✓  GIF saved as scan_phi.gif")
