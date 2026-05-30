import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.vipa_focus import PARAMS_80, vipa_rays
from src.crosssections import crosssection_xz_naive

if __name__ == "__main__":
    params = PARAMS_80

    EXTENT_Z = 20e-3
    NZ = 2000

    rays = vipa_rays(params)
    z_scan, xf, profiles = crosssection_xz_naive(
        rays, params, extent_z=EXTENT_Z, n_z=NZ
    )
    print(f"profiles shape: {profiles.shape}")
    np.savez(ROOT / "data/vipa_focus_xz.npz", z_scan=z_scan, xf=xf, profiles=profiles)
