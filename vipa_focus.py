import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from core import *
import time
import imageio
from crosssections import *


def vipa_rays(
    params: dict,
) -> np.ndarray:
    """
    Complex electric field in the vipa output plane.
    """
    Nx = params["Nx"]
    Ny = params["Ny"]
    lx = params["lx"]
    ly = params["ly"]
    FSR_Ratio = params["FSR_Ratio"]
    phi = params["phi"]
    w = params["w"]
    d = params["d"]
    phase_amp_func = params.get("phase_amp_func", None)
    displacement_func = params.get("displacement_func", None)
    phase_correction = params.get("phase_correction", None)
    #

    rays = []
    for nx in range(Nx):
        for ny in range(Ny):
            ix = nx - (Nx - 1) / 2
            iy = ny - (Ny - 1) / 2
            center_x = (ix) * d
            center_y = (iy) * d
            if displacement_func is not None:
                dx, dy = displacement_func(ix, iy, nx, ny)
                center_x += dx
                center_y += dy
            intensity = np.exp(-(nx * lx + ny * ly))
            phase = -(ix * FSR_Ratio + iy) * (phi)
            if phase_correction is not None:
                # pass
                phase += phase_correction[nx, ny]
            #
            ray = {
                "x": center_x,
                "y": center_y,
                "w": w,
                "ix": ix,
                "iy": iy,
                "nx": nx,
                "ny": ny,
                "intensity": intensity,
                "phase": phase,
            }
            if phase_amp_func is not None:
                phase_amp_func_i = (
                    lambda Xi, Yi, ix=ix, iy=iy, nx=nx, ny=ny: phase_amp_func(
                        ix, iy, nx, ny, Xi, Yi
                    )
                )
                ray["phase_amp_func"] = phase_amp_func_i
            rays.append(ray)
    #
    return rays


def rays_from_file(
    filename: str,
    params: dict,
) -> np.ndarray:
    data = np.load(filename, allow_pickle=True)
    #
    FSR_Ratio = params["FSR_Ratio"]
    phi = params["phi"]
    #
    xList = data["xList"] * 1e-2  # optable unit is cm
    yList = data["yList"] * 1e-2
    # center the beams
    xList -= np.mean(xList)
    yList -= np.mean(yList)
    tXList = data["tXList"]
    tYList = data["tYList"]
    tXList -= np.mean(tXList)
    tYList -= np.mean(tYList)
    IList = data["IList"]
    print(xList.shape, yList.shape, tXList.shape, tYList.shape, IList.shape)
    #
    Nx = len(xList)
    Ny = 1
    assert (
        len(xList) == Nx * Ny
    ), f"Number of traced rays {len(xList)} does not match Nx*Ny={Nx*Ny}"
    print(f"Number of traced rays: {len(xList)} matching Nx*Ny={Nx*Ny}")
    #
    wl = params["lambda"]

    def tilt_phase(tx, ty, Xi, Yi):
        k = 2 * np.pi / wl
        return (tx * k) * Xi + (ty * k) * Yi, np.ones_like(Xi)

    #
    rays = []  # for elec2d format
    for ny in range(Ny):
        for nx in range(Nx):
            idx = ny * Nx + nx
            ix = nx - (Nx - 1) / 2
            iy = ny - (Ny - 1) / 2
            intensity = IList[idx]
            phase = -(ix * FSR_Ratio + iy) * (phi)
            phase_amp_func_i = lambda Xi, Yi, tx=tXList[idx], ty=tYList[
                idx
            ]: tilt_phase(tx, ty, Xi, Yi)
            ray = {
                "x": xList[idx],
                "y": yList[idx],
                "w": 61e-6,
                "ix": ix,
                "iy": iy,
                "nx": nx,
                "ny": ny,
                "intensity": intensity,
                "phase": phase,
                "phase_amp_func": phase_amp_func_i,
            }
            rays.append(ray)

    # field = rays2elec2d(Xi, Yi, rays, params, **kwargs)
    # return field
    return rays


DSP = -100e-6  # 20 um


def misaligned_tilt(ix, iy, nx, ny, Xi, Yi):
    tx = -np.cos(2 * np.pi * nx / 4) * (DSP / 46.7e-3)
    return (tx * (2 * np.pi / 780e-9)) * Xi, np.ones_like(Xi)


def misaligned_displacement(ix, iy, nx, ny):
    # dx = -np.sin(2 * np.pi * nx / 4) * DSP
    # dy = 0
    dx = 0
    dy = -np.sin(2 * np.pi * ny / 4) * DSP
    return dx, dy


PARAMS_100 = {
    "Nx": 100,
    "Ny": 100,
    "FSR_Ratio": 100,
    "lx": 0.01,
    "ly": 0.01,
    "w": 61e-6,  # beam waist
    "d": 360e-6,  # beam spacing
    "f": 0.2,
    "phi": 0.0,
    "lambda": 780e-9,
    "D": 40e-2,  # real space extent
    "RESOLUTION_X": 25e-6,  # real space resolution
    "extent_f": 1000e-6,  # focal plane extent, only for plotting
    # "phase_amp_func": misaligned_tilt,
    # "displacement_func": misaligned_displacement,
    # "gouy_phase": 0.0,
}
PARAMS_10 = {
    "Nx": 1,
    "Ny": 9,
    "FSR_Ratio": 22.0,
    "Lrt": 2.311,
    "lx": 0.198,
    "ly": 0.052,
    "w": 108e-6,  # beam waist
    "d": 1000e-6,  # beam spacing
    "f": 0.2,
    "phi": 0.0,
    "lambda": 780e-9,
    "D": 10e-2,  # real space extent
    "RESOLUTION_X": 25e-6,  # real space resolution
    "extent_x": 1e-2,
    "extent_f": 800e-6,  # focal plane extent, only for plotting
    # "phase_amp_func": misaligned_tilt,
    # "displacement_func": misaligned_displacement,
    "zfi": None,
    # "gouy_phase": 0.0,
}

if __name__ == "__main__":

    TYPE = 5

    # params = PARAMS_100
    params = PARAMS_10

    if TYPE == 0:
        """XY focal plane profile"""
        zf = 0
        # phase_correction = np.load("./data/vipa_fitted_phases_phi_0.00.npy")
        # params["phase_correction"] = np.array(phase_correction)
        rays = vipa_rays(params)

        xf, yf, E_tilde_0, intensity = crosssection_xy(
            rays, params, zf=0e-6, show_E_field=False, show_focus=False
        )
        print(xf[1] - xf[0])
        # np.savez("./data/vipa_100.npz", xf=xf, yf=yf, intensity=intensity)
        # np.save("./data/vipa_focus_demo_1d.npy", intensity)
        linecut = intensity[:, intensity.shape[1] // 2]

        params.update({"phi": 0.5})
        rays = vipa_rays(params)

        xf, yf, E_tilde_0, intensity = crosssection_xy(
            rays, params, zf=0e-6, show_E_field=False, show_focus=False
        )
        linecut_phase_shifted = intensity[:, intensity.shape[1] // 2]

        params.update({"phase_amp_func": misaligned_tilt, "phi": 0})
        rays = vipa_rays(params)

        xf, yf, E_tilde_0, intensity = crosssection_xy(
            rays, params, zf=0e-6, show_E_field=False, show_focus=False
        )
        linecut_misaligned_tilt_only = intensity[:, intensity.shape[1] // 2]

        params.update(
            {"phase_amp_func": None, "displacement_func": misaligned_displacement}
        )
        rays = vipa_rays(params)

        xf, yf, E_tilde_0, intensity = crosssection_xy(
            rays, params, zf=0e-6, show_E_field=False, show_focus=False
        )
        linecut_misaligned_displacement = intensity[:, intensity.shape[1] // 2]

        params.update({"phi": 0.5})
        rays = vipa_rays(params)
        xf, yf, E_tilde_0, intensity = crosssection_xy(
            rays, params, zf=0e-6, show_E_field=False, show_focus=False
        )
        linecut_disp_phase_shifted = intensity[:, intensity.shape[1] // 2]

        plt.figure()
        plt.plot(xf * 1e6, linecut, "b-", label="Aligned")
        plt.plot(
            xf * 1e6, linecut_phase_shifted, "orange", label="Phase shifted (0.5 rad)"
        )
        plt.plot(
            xf * 1e6, linecut_misaligned_tilt_only, "r-", label="Misaligned tilt only"
        )
        plt.plot(
            xf * 1e6,
            linecut_misaligned_displacement,
            "g-",
            label="Misaligned displacement only",
        )
        plt.plot(
            xf * 1e6,
            linecut_disp_phase_shifted,
            "k--",
            label="Misalignment disp Phase shifted (0.5 rad)",
        )
        plt.legend()

        # plt.figure()
        # plt.plot(xf * 1e6, linecut, "b-", label="Aligned")
        # plt.xlabel(r"$x_f$ (µm)")
        # plt.ylabel("Intensity (arb.)")
        # plt.title(f"Focal plane intensity linecut at y=0 µm")
        # plt.tight_layout()
        plt.yscale("log")
        plt.show()
        # np.save("./data/vipa_focus_demo_linecut.npy", linecut)

    elif TYPE in [1, 2]:
        """XZ profile"""
        EXTENT_Z = 20e-3
        NZ = 2000
        if TYPE == 1:
            rays = vipa_rays(params)
            z_scan, xf, profiles = crosssection_xz(
                rays, params, extent_z=EXTENT_Z, n_z=NZ
            )
            print(f"profiles shape: {profiles.shape}")
            np.savez(
                "./data/vipa_focus_xz.npz", z_scan=z_scan, xf=xf, profiles=profiles
            )

        elif TYPE == 2:
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
            np.save("./data/vipa_focus_xz_phi_scan.npy", gif_data)
            gif_data = (gif_data * 255).astype(np.uint8)
            # kron the gif to make xz aspect equal
            dx = xf[1] - xf[0]
            dz = z_scan[1] - z_scan[0]
            ratio = dz / dx
            print(f"dx={dx*1e6:.2f} um, dz={dz*1e6:.2f} um, ratio={ratio:.2f}")
            # gif_data = np.kron(gif_data, np.ones((1, 1, int(ratio)), dtype=np.uint8))
            # ----- write the animated GIF -----------------------------------------
            imageio.mimsave("./figs/scan_phi.gif", gif_data, fps=5, loop=0)
            print("✓  GIF saved as scan_phi.gif")

    elif TYPE == 3:
        _, _, _, intensity = crosssection_xy(params, zf=0)
        H, W = intensity.shape
        data = np.zeros((11, 11, H, W))

        for j in range(11):
            for i in range(11):
                print(f"Calculating for (i,j)=({i},{j})")
                params["phi"] = (i / 11 + 2 * j) * np.pi / 11
                xf, yf, _, intensity = crosssection_xy(
                    params,
                    zf=0e-6,
                    show_focus=False,
                )
                data[i, j, :, :] = intensity

    elif TYPE == 4:
        dsps = np.linspace(-200e-6, 200e-6, 11)
        gifs = []
        for dsp in dsps:
            DSP = dsp
            rays = vipa_rays(params)
            _, intensity = crosssection_x(
                rays,
                params,
                zf=0e-6,
                show_focus=False,
            )
            plt.figure()
            plt.plot(intensity)
            plt.ylim(0, 50e15)
            plt.show()
        gifs = np.array(gifs)
        gifs = (gifs / np.max(gifs) * 255).astype(np.uint8)
        imageio.mimsave("./figs/vipa_misalignment_dsp.gif", gifs, fps=5, loop=0)
        print("✓  GIF saved as vipa_misalignment_dsp.gif")

    elif TYPE == 5:
        """Rays from file"""
        rays = rays_from_file("./data/optable/ripa_gen2_2nd_mon0_rays.npz", params)
        xf, yf, E_tilde_0, intensity = crosssection_xy(
            rays, params, zf=0e-6, show_E_field=False, show_focus=True, log_scale=False
        )
