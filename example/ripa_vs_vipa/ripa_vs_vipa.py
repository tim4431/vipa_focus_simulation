import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.vipa_focus import (
    PARAMS_10,
)
from src.crosssections import crosssection_xy

if __name__ == "__main__":
    params = PARAMS_10

    vipaRays = []
    ripaRays = []
    #
    Ny = 9
    phi = params["phi"]
    w0 = params["w"]
    wl = params["lambda"]
    zR = np.pi * w0**2 / wl
    Lrt = 0.094  # first rt
    print(Lrt / zR)

    def spherical_phase(R, phi_gouy, Xi, Yi):
        k = 2 * np.pi / wl
        return -k * (Xi**2 + Yi**2) / (2 * R) + phi_gouy, np.ones_like(Xi)

    #

    for ny in range(Ny):
        idx = ny
        iy = ny - (Ny - 1) / 2
        phase = -(iy) * (phi)
        z = ny * Lrt
        R = z + zR**2 / z if z != 0 else np.inf
        phi_gouy = np.arctan(z / zR)
        vipa_phase_amp_func_i = lambda Xi, Yi, R=R: spherical_phase(R, phi_gouy, Xi, Yi)
        riparay = {
            "x": 0,
            "y": params["d"] * iy,
            "w": w0,
            "iy": iy,
            "ny": ny,
            "intensity": 1,
            "phase": phase,
            "gouy_phase": 0.0,
        }
        viparay = {
            "x": 0,
            "y": params["d"] * iy,
            "w": w0 * np.sqrt(1 + (z / zR) ** 2),
            "iy": iy,
            "ny": ny,
            "intensity": 1,
            "phase": phase,
            "gouy_phase": 0.0,
            "phase_amp_func": vipa_phase_amp_func_i,
        }
        ripaRays.append(riparay)
        vipaRays.append(viparay)

    DEBUG = False
    # DEBUG = True
    xf, yf, E_tilde_0, intensity = crosssection_xy(
        vipaRays, params, zf=0e-6, show_E_field=DEBUG, show_focus=DEBUG
    )

    xf, yf, E_tilde_0_ripa, intensity_ripa = crosssection_xy(
        ripaRays, params, zf=0e-6, show_E_field=DEBUG, show_focus=DEBUG
    )

    ix0 = np.argmin(np.abs(xf))
    profile_vipa = intensity[:, ix0]
    profile_ripa = intensity_ripa[:, ix0]

    dxf = xf[1] - xf[0]
    dyf = yf[1] - yf[0]
    energy_vipa = np.sum(intensity) * dxf * dyf
    energy_ripa = np.sum(intensity_ripa) * dxf * dyf
    print(f"VIPA total energy (full FOV): {energy_vipa:.6e}")
    print(f"RIPA total energy (full FOV): {energy_ripa:.6e}")
    print(f"VIPA / RIPA ratio           : {energy_vipa / energy_ripa:.4f}")

    per_ray_profiles = []
    for ray in vipaRays:
        _, yf_i, _, intensity_i = crosssection_xy(
            [ray], params, zf=0e-6, show_E_field=False, show_focus=False
        )
        ix0_i = np.argmin(np.abs(xf))
        per_ray_profiles.append((ray["ny"], yf_i, intensity_i[:, ix0_i]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(yf * 1e6, profile_vipa, label="VIPA (sum)", color="C0", lw=2)
    axes[0].plot(yf * 1e6, profile_ripa, label="RIPA (sum)", color="C1", lw=2)
    axes[0].set_xlabel(r"$y_f$ ($\mu$m)")
    axes[0].set_ylabel("Intensity (arb.)")
    axes[0].set_title("Linear scale")
    axes[0].legend()

    axes[1].semilogy(yf * 1e6, profile_vipa, label="VIPA (sum)", color="C0", lw=2)
    axes[1].semilogy(yf * 1e6, profile_ripa, label="RIPA (sum)", color="C1", lw=2)
    axes[1].set_xlabel(r"$y_f$ ($\mu$m)")
    axes[1].set_ylabel("Intensity (arb., log)")
    axes[1].set_title("Log scale")
    axes[1].legend()

    fig.suptitle("RIPA vs VIPA focal-plane linecut ($x_f \\approx 0$)")
    plt.tight_layout()

    cmap = plt.get_cmap("viridis")
    n_rays = len(per_ray_profiles)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for k, (ny_k, yf_k, prof_k) in enumerate(per_ray_profiles):
        c = cmap(k / max(n_rays - 1, 1))
        axes2[0].plot(yf_k * 1e6, prof_k, color=c, lw=1.0, label=f"ny={ny_k}")
        axes2[1].semilogy(yf_k * 1e6, prof_k, color=c, lw=1.0, label=f"ny={ny_k}")

    axes2[0].set_xlabel(r"$y_f$ ($\mu$m)")
    axes2[0].set_ylabel("Intensity (arb.)")
    axes2[0].set_title("Linear scale")
    axes2[0].legend(fontsize=8, ncol=2)

    axes2[1].set_xlabel(r"$y_f$ ($\mu$m)")
    axes2[1].set_ylabel("Intensity (arb., log)")
    axes2[1].set_title("Log scale")
    axes2[1].legend(fontsize=8, ncol=2)

    fig2.suptitle("Per-ray VIPA focal-plane linecut ($x_f \\approx 0$)")
    plt.tight_layout()
    plt.show()

    exit(0)
