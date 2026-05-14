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
    PARAMS_100,
    PARAMS_10,
    vipa_rays,
    misaligned_tilt,
    misaligned_displacement,
)
from src.crosssections import crosssection_xy

if __name__ == "__main__":
    # params = PARAMS_100
    params = PARAMS_80
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
    # # before saving, crop the intensity data into params["extent_f"]
    # extent_f = params["extent_f"]
    # mask_x = np.abs(xf) <= extent_f / 2
    # mask_y = np.abs(yf) <= extent_f / 2
    # mask = np.outer(mask_y, mask_x)
    # xf = xf[mask_x]
    # yf = yf[mask_y]
    # intensity = intensity[mask].reshape(len(yf), len(xf))
    #
    print(xf[1] - xf[0])
    print(intensity.shape)
    # np.savez(ROOT / "data/vipa_100.npz", xf=xf, yf=yf, intensity=intensity)
    # np.savez(ROOT / "data/vipa_focus_demo_80.npz", xf=xf, yf=yf, intensity=intensity)
    # np.save(ROOT / "data/vipa_focus_demo_1d.npy", intensity)

    exit(0)
    # linecut = intensity[:, intensity.shape[1] // 2]

    # params.update({"phi": 0.5})
    # rays = vipa_rays(params)

    # xf, yf, E_tilde_0, intensity = crosssection_xy(
    #     rays, params, zf=0e-6, show_E_field=False, show_focus=False
    # )
    # linecut_phase_shifted = intensity[:, intensity.shape[1] // 2]

    # params.update({"phase_amp_func": misaligned_tilt, "phi": 0})
    # rays = vipa_rays(params)

    # xf, yf, E_tilde_0, intensity = crosssection_xy(
    #     rays, params, zf=0e-6, show_E_field=False, show_focus=False
    # )
    # linecut_misaligned_tilt_only = intensity[:, intensity.shape[1] // 2]

    # params.update(
    #     {"phase_amp_func": None, "displacement_func": misaligned_displacement}
    # )
    # rays = vipa_rays(params)

    # xf, yf, E_tilde_0, intensity = crosssection_xy(
    #     rays, params, zf=0e-6, show_E_field=False, show_focus=False
    # )
    # linecut_misaligned_displacement = intensity[:, intensity.shape[1] // 2]

    # params.update({"phi": 0.5})
    # rays = vipa_rays(params)
    # xf, yf, E_tilde_0, intensity = crosssection_xy(
    #     rays, params, zf=0e-6, show_E_field=False, show_focus=False
    # )
    # linecut_disp_phase_shifted = intensity[:, intensity.shape[1] // 2]

    # plt.figure()
    # plt.plot(xf * 1e6, linecut, "b-", label="Aligned")
    # plt.plot(
    #     xf * 1e6, linecut_phase_shifted, "orange", label="Phase shifted (0.5 rad)"
    # )
    # plt.plot(
    #     xf * 1e6, linecut_misaligned_tilt_only, "r-", label="Misaligned tilt only"
    # )
    # plt.plot(
    #     xf * 1e6,
    #     linecut_misaligned_displacement,
    #     "g-",
    #     label="Misaligned displacement only",
    # )
    # plt.plot(
    #     xf * 1e6,
    #     linecut_disp_phase_shifted,
    #     "k--",
    #     label="Misalignment disp Phase shifted (0.5 rad)",
    # )
    # plt.legend()

    # # plt.figure()
    # # plt.plot(xf * 1e6, linecut, "b-", label="Aligned")
    # # plt.xlabel(r"$x_f$ (µm)")
    # # plt.ylabel("Intensity (arb.)")
    # # plt.title(f"Focal plane intensity linecut at y=0 µm")
    # # plt.tight_layout()
    # plt.yscale("log")
    # plt.show()
    # # np.save(ROOT / "data/vipa_focus_demo_linecut.npy", linecut)
