"""XY focal plane profile (was vipa_focus.py TYPE == 0)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.vipa_focus import (
    PARAMS_80,
    PARAMS_80_TWZ,
    PARAMS_100,
    PARAMS_10,
    PARAMS_10_TWZ,
    vipa_rays,
)
from src.crosssections import crosssection_xy

if __name__ == "__main__":
    # params = PARAMS_100
    # params = PARAMS_10_TWZ
    params = PARAMS_80_TWZ
    # params = PARAMS_10

    FSR_L = params["lambda"] * params["f"] / params["d"]
    print("FSR_L =", FSR_L)
    # exit(0)
    zf = 0
    # phase_correction = np.load(ROOT / "data/vipa_fitted_phases_phi_0.00.npy")
    # params["phase_correction"] = np.array(phase_correction)
    rays = vipa_rays(params)

    xf, yf, E_tilde_0, intensity = crosssection_xy(
        rays, params, zf=0e-6, show_E_field=False, show_focus=True
    )

    exit(0)
    #

    # # before saving, crop the intensity data into params["extent_f"]
    extent_f = params["extent_f"]
    mask_x = np.abs(xf) <= extent_f / 2
    mask_y = np.abs(yf) <= extent_f / 2
    mask = np.outer(mask_y, mask_x)
    xf = xf[mask_x]
    yf = yf[mask_y]
    intensity = intensity[mask].reshape(len(yf), len(xf))
    #
    print(xf[1] - xf[0])
    print(intensity.shape)
    # np.savez(ROOT / "data/vipa_100.npz", xf=xf, yf=yf, intensity=intensity)
    # np.savez(ROOT / "data/vipa_focus_demo_80.npz", xf=xf, yf=yf, intensity=intensity)
    # np.save(ROOT / "data/vipa_focus_demo_1d.npy", intensity)
    exit(0)
