"""Phi scan simulation data in the focal plane (was vipa_focus.py TYPE == 3)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.vipa_focus import PARAMS_80, vipa_rays
from src.crosssections import crosssection_xy


if __name__ == "__main__":
    params = PARAMS_80

    rays = vipa_rays(params)
    _, _, _, intensity = crosssection_xy(
        rays, params, zf=0, show_E_field=True, show_focus=True
    )
    H, W = intensity.shape
    print(H, W)
    N = 80
    data = np.zeros((N, N, H, W))

    _cnt = 0
    for j in range(N):
        for i in range(N):
            # if _cnt >= 3:
            #     break
            # print(_cnt)
            # _cnt += 1
            print(f"Calculating for (i,j)=({i},{j})")
            # params["phi"] = (i / N + 2 * j) * np.pi / N
            params["phi"] = (i / N + j) * (2 * np.pi) / N
            rays = vipa_rays(params)
            xf, yf, _, intensity = crosssection_xy(
                rays,
                params,
                zf=0e-6,
                show_focus=False,
            )
            data[i, j, :, :] = intensity

    np.save(ROOT / "data/vipa_focus_demo_80.npy", data)
