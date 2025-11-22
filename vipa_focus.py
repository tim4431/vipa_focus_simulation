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
            #
            ray = {
                "x": center_x,
                "y": center_y,
                "w": w,
                "ix": ix,
                "iy": iy,
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
    params: dict,
) -> np.ndarray:
    FSR_Ratio = params["FSR_Ratio"]
    #
    # FILENAME = "./ripa_gen2_1st_mon0_rays.npz"
    FILENAME = "./ripa_gen2_2nd_mon0_rays.npz"
    data = np.load(FILENAME, allow_pickle=True)
    xList = data["xList"] * 1e-2
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
                "intensity": intensity,
                "phase": phase,
                "phase_amp_func": phase_amp_func_i,
            }
            rays.append(ray)

    # field = rays2elec2d(Xi, Yi, rays, params, **kwargs)
    # return field
    return rays


DSP = -10e-6  # 20 um


def misaligned_tilt(ix, iy, nx, ny, Xi, Yi):
    tx = -np.cos(2 * np.pi * nx / 4) * (DSP / 46.7e-3)
    return (tx * (2 * np.pi / 780e-9)) * Xi, np.ones_like(Xi)


def misaligned_displacement(ix, iy, nx, ny):
    dx = -np.sin(2 * np.pi * nx / 4) * DSP
    dy = 0
    return dx, dy


# ----------------------------------------------------------------------
# Self-test / demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    PARAMS_100 = {
        "Nx": 100,
        "Ny": 100,
        "FSR_Ratio": 100,
        "lx": 0.01,
        "ly": 0.01,
        "w": 108e-6,  # beam waist
        "d": 300e-6,  # beam spacing
        "f": 0.2,
        "phi": 0.0,
        "lambda": 780e-9,
        "D": 20e-2,  # real space extent
        "RESOLUTION_X": 25e-6,  # real space resolution
        "extent_f": 1000e-6,  # focal plane extent, only for plotting
        "phase_amp_func": misaligned_tilt,
        "displacement_func": misaligned_displacement,
    }
    PARAMS_10 = {
        "Nx": 8,
        # "Nx": 1,
        "Ny": 9,
        # "Ny": 1,
        "FSR_Ratio": 22.0,
        "Lrt": 2.311,
        "lx": 0,
        # "lx": 0.15,
        "ly": 0.052,
        "w": 108e-6,  # beam waist
        "d": 1000e-6,  # beam spacing
        "f": 0.2,
        "phi": 0.0,
        "lambda": 780e-9,
        "D": 10e-2,  # real space extent
        "RESOLUTION_X": 25e-6,  # real space resolution
        "extent_x": 1e-2,
        "extent_f": 500e-6,  # focal plane extent, only for plotting
        "phase_amp_func": misaligned_tilt,
        "displacement_func": misaligned_displacement,
        "zfi": None,
    }

    TYPE = 0

    params = PARAMS_100

    if TYPE == 0:
        phi = 0
        params.update({"phi": phi})
        zf = 0
        rays = vipa_rays(params)

        _, _, E_tilde_0, intensity = crosssection_xy(
            rays, params, zf=0e-6, show_E_field=True, show_focus=True
        )
        # H, W = intensity.shape
        # gif_data = []
        # for D in np.linspace(-100e-6, 100e-6, 21):
        #     DSP = D
        #     # DSP = 100e-6  # 20 um
        #     _, _, E_tilde_0, intensity = crosssection_xy(
        #         params, zf=0e-6, show_E_field=False, show_focus=False
        #     )
        #     gif_data.append(intensity)
        # gif_data = np.array(gif_data)  # shape (ND, n_xf
        # gif_data = gif_data / np.max(gif_data)
        # gif_data = (gif_data * 255).astype(np.uint8)
        # imageio.mimsave("./figs/scan_displacement.gif", gif_data, fps=5, loop=0)

    elif TYPE in [1, 2]:
        EXTENT_Z = 5e-3
        NZ = 200
        if TYPE == 1:
            z_scan, xf, profiles = crosssection_xz(params, extent_z=EXTENT_Z, n_z=NZ)

        elif TYPE == 2:
            NPHI = 30
            gif_data = []
            for phi in np.linspace(0.0, 2 * np.pi, NPHI):
                print(f"phi = {phi:.2f}")
                params["phi"] = phi  # update phase

                z_scan, xf, profiles = crosssection_xz(
                    params, extent_z=EXTENT_Z, n_z=NZ, show_focus=False
                )
                gif_data.append(profiles)
            gif_data = np.array(gif_data)  # shape (NPHI, n_xf, n_z)
            # normalize to [0,255]
            gif_data = gif_data / np.max(gif_data)
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
                # print(intensity.shape)
                data[i, j, :, :] = intensity
                # plt.imshow(
                #     intensity,
                #     extent=[-params["extent_f"] * 1e6, params["extent_f"] * 1e6] * 2,
                #     origin="lower",
                #     cmap="rainbow",
                #     aspect="equal",
                #     vmin=0,
                # )
                # plt.title(
                #     f"(i,j) = ({i}, {j}), phi = {(i / 11 + 2 * j) * np.pi / 11:.2f} rad"
                # )
                # plt.xlabel(r"$x_f$ (µm)")
                # plt.ylabel(r"$y_f$ (µm)")
                # plt.tight_layout()
                # plt.savefig(f"./figs/vipa_focus_demo_phi_{i}_{j}.png")

        # np.save("vipa_focus_demo_data.npy", data)

    #
