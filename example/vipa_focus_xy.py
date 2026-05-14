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
    PARAMS_10_TWZ,
    vipa_rays,
    misaligned_tilt,
    misaligned_displacement,
)
from src.crosssections import crosssection_xy
from scipy.optimize import least_squares

if __name__ == "__main__":
    # params = PARAMS_100
    params = PARAMS_10_TWZ
    # params = PARAMS_10

    params["phi"] = (8.2 / 11 + 2 * 8) * np.pi / 11

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

    # --- 2D Gaussian fit to extract beam waist ---
    # Parameterize by (x0, y0, sigma_x, sigma_y, theta, amp, offset) — always PSD,
    # avoids the inf/NaN failure mode of fitting covariance-matrix entries directly.
    X2d, Y2d = np.meshgrid(xf, yf)
    I_norm = intensity / intensity.max()

    def gauss2d_rot(x, y, x0, y0, sx, sy, theta, amp, off):
        ct, st = np.cos(theta), np.sin(theta)
        xr = (x - x0) * ct + (y - y0) * st
        yr = -(x - x0) * st + (y - y0) * ct
        return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + off

    # Initial guess from intensity-weighted moments
    Iw = np.clip(I_norm - I_norm.min(), 0, None)
    Iw_sum = Iw.sum()
    x0_0 = float((X2d * Iw).sum() / Iw_sum)
    y0_0 = float((Y2d * Iw).sum() / Iw_sum)
    sx_0 = float(np.sqrt(((X2d - x0_0) ** 2 * Iw).sum() / Iw_sum))
    sy_0 = float(np.sqrt(((Y2d - y0_0) ** 2 * Iw).sum() / Iw_sum))
    p0 = [x0_0, y0_0, sx_0, sy_0, 0.0, 1.0, 0.0]

    def _resid(p):
        return (gauss2d_rot(X2d, Y2d, *p) - I_norm).ravel()

    lb = [xf[0], yf[0], 1e-9, 1e-9, -np.pi, 0.0, -1.0]
    ub = [xf[-1], yf[-1], (xf[-1] - xf[0]), (yf[-1] - yf[0]), np.pi, 10.0, 1.0]
    res = least_squares(_resid, p0, bounds=(lb, ub), ftol=1e-10, xtol=1e-10)
    x0_f, y0_f, sx_f, sy_f, th_f, amp_f, off_f = res.x

    # intensity I = I0 * exp(-2 r^2 / w^2)  =>  sigma = w/2  =>  w = 2 sigma
    w_a = 2 * sx_f
    w_b = 2 * sy_f
    print(f"Fit center: ({x0_f * 1e6:.3f}, {y0_f * 1e6:.3f}) um")
    print(f"Rotation angle: {np.degrees(th_f):.2f} deg")
    print(f"Principal-axis waists (1/e^2): w_a = {w_a * 1e6:.3f} um, w_b = {w_b * 1e6:.3f} um")

    Z_fit = gauss2d_rot(X2d, Y2d, x0_f, y0_f, sx_f, sy_f, th_f, amp_f, off_f)
    Z_fit = Z_fit * intensity.max()
    fig, ax = plt.subplots()
    im = ax.imshow(
        intensity,
        origin="lower",
        extent=[xf[0] * 1e6, xf[-1] * 1e6, yf[0] * 1e6, yf[-1] * 1e6],
        cmap="inferno",
    )
    levels = intensity.max() * np.array([np.exp(-2), np.exp(-1), 0.5])
    ax.contour(X2d * 1e6, Y2d * 1e6, Z_fit, levels=levels, colors="cyan", linewidths=1)
    ax.set_xlabel(r"$x_f$ (um)")
    ax.set_ylabel(r"$y_f$ (um)")
    ax.set_title(
        f"Gaussian fit: w_a = {w_a * 1e6:.2f} um, w_b = {w_b * 1e6:.2f} um, "
        f"theta = {np.degrees(th_f):.1f} deg"
    )
    plt.colorbar(im, ax=ax, label="Intensity (arb.)")
    plt.tight_layout()
    plt.show()

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
