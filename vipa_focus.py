import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from elec import *
import time
import imageio
from sequences import *


def vipa_rays(
    params: dict,
    **kwargs,
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
    **kwargs,
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


def crosssection_xy(
    params: dict,
    zf: float = 0.0,
    show_focus: bool = True,
    show_E_field: bool = False,
    **kwargs,
):
    """
    Special-case: square patch (default ±500 µm) with uniform resolution.
    """
    D = params["D"]
    RESOLUTION_X = params["RESOLUTION_X"]
    N_grid = int(D / (2 * RESOLUTION_X)) * 2 + 1
    xi = np.linspace(-D / 2, D / 2, N_grid)
    yi = np.linspace(-D / 2, D / 2, N_grid)
    rays = vipa_rays(params, **kwargs)
    # rays = rays_from_file(params, **kwargs)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    # print(Xi.shape, Yi.shape)

    Ei = rays2elec2d(Xi, Yi, rays, params, **kwargs)

    zfi = params.get("zfi", None)
    if zfi is not None:
        print(f"Propagating initial zf = {zfi*1e3:.2f} mm")
        Ei = freespace_propagation(xi, yi, Ei, params["lambda"], zfi)
    if show_E_field:
        extent = [-D / 2 * 1e6, D / 2 * 1e6, -D / 2 * 1e6, D / 2 * 1e6]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Intensity subplot
        im1 = axes[0].imshow(
            np.abs(Ei) ** 2,
            extent=extent,
            origin="lower",
            cmap="rainbow",
            aspect="equal",
            vmin=0,
        )
        axes[0].set_xlabel(r"$x$ (source plane, $\mu$m)")
        axes[0].set_ylabel(r"$y$ (source plane, $\mu$m)")
        axes[0].set_title("Source plane intensity")
        plt.colorbar(im1, ax=axes[0], label="Intensity (arb.)")

        # Phase subplot
        im2 = axes[1].imshow(
            np.angle(Ei) + np.abs(Ei) ** 2 / np.max(np.abs(Ei) ** 2),
            extent=extent,
            origin="lower",
            cmap="Reds",
            aspect="equal",
        )
        axes[1].set_xlabel(r"$x$ (source plane, $\mu$m)")
        axes[1].set_ylabel(r"$y$ (source plane, $\mu$m)")
        axes[1].set_title("Source plane phase")
        plt.colorbar(im2, ax=axes[1], label="Phase (rad)")

        plt.tight_layout()
        plt.show()
    (xf, yf), E_tilde = calc_field_after_lens(
        xi,
        yi,
        Ei,
        wl=params["lambda"],
        f=params["f"],
        zf=zf,
        **kwargs,
    )
    if params.get("pinhole"):
        print("Applying pinhole...")
        pinhole_dict = params.get("pinhole")
        pinhole_diameter = pinhole_dict.get("diameter", 5e-6)
        pinhole_x = pinhole_dict.get("x", 0.0)
        pinhole_y = pinhole_dict.get("y", 0.0)
        mask_pinhole = np.zeros_like(E_tilde, dtype=bool)
        Xf, Yf = np.meshgrid(xf, yf, indexing="xy")
        mask_pinhole[
            np.sqrt((Xf - pinhole_x) ** 2 + (Yf - pinhole_y) ** 2)
            <= pinhole_diameter / 2
        ] = True
        E_tilde *= mask_pinhole
        # propagate by zf_pinhole
        zf_pinhole = pinhole_dict.get("zf_pinhole", 0.0e-3)
        print(f"Propagating by zf_pinhole = {zf_pinhole*1e3:.2f} mm")
        E_tilde = freespace_propagation(xf, yf, E_tilde, params["lambda"], zf_pinhole)
    #
    extent_f = params["extent_f"]
    # crop data to ±extent_f
    xf_mask = np.abs(xf) < extent_f
    yf_mask = np.abs(yf) < extent_f
    xf = xf[xf_mask]
    yf = yf[yf_mask]
    E_tilde = E_tilde[np.ix_(yf_mask, xf_mask)]
    intensity = np.abs(E_tilde) ** 2

    if show_focus:
        # plot
        extent_um = extent_f * 1e6
        plt.figure(figsize=(6, 5))
        plt.imshow(
            intensity,
            extent=[-extent_um, extent_um, -extent_um, extent_um],
            origin="lower",
            cmap="rainbow",
            aspect="equal",
            vmin=0,
        )
        plt.colorbar(label="Intensity (arb.)")
        # ticks = np.arange(-extent_um, extent_um + 1e-12, 100)
        # plt.xticks(ticks, ticks.astype(int))
        # plt.yticks(ticks, ticks.astype(int))
        plt.xlabel(r"$x_f$ (focal plane, $\mu$m)")
        plt.ylabel(r"$y_f$ (focal plane, $\mu$m)")
        plt.title(rf"Interference pattern (zf = {zf*1e6:.1f} µm)")
        plt.tight_layout()
        plt.show()

    return xf, yf, E_tilde, intensity


def crosssection_x(
    params: dict,
    zf: float = 0.0,
    show_focus: bool = False,
    **kwargs,
):
    """
    Return |E|² at Y_f≈0 as a function of x_f.

    Returns
    -------
    xf       : 1-D array (metres)
    profile  : 1-D array  |E|²(x_f)   shape == len(xf)
    """

    D = params["D"]
    RESOLUTION_X = params["RESOLUTION_X"]
    N_grid = int(D / (2 * RESOLUTION_X)) * 2 + 1
    xi = np.linspace(-D / 2, D / 2, N_grid)
    yi = np.array([0])

    rays = vipa_rays(params, **kwargs)
    # rays = rays_from_file(params, **kwargs)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    Ei = rays2elec2d(Xi, Yi, rays, params, **kwargs)
    zfi = params.get("zfi", None)
    if zfi is not None:
        print(f"Propagating initial zf = {zfi*1e3:.2f} mm")
        Ei = freespace_propagation(xi, yi, Ei, params["lambda"], zfi)
    (xf, yf), profile = calc_field_after_lens(
        xi, yi, Ei, wl=params["lambda"], f=params["f"], zf=zf, **kwargs
    )
    profile = np.abs(profile) ** 2  # shape (1, len(yf))
    # print(profile.shape)
    profile = profile.flatten()  # shape (len(yf),)

    extent_f = params["extent_f"]
    xf_mask = np.abs(xf) < extent_f
    xf = xf[xf_mask]
    profile = profile[xf_mask]

    if show_focus:
        plt.figure()
        plt.plot(xf * 1e6, profile)
        plt.xlabel(r"$x_f$ (µm)")
        # plt.xlim(-extent_f * 1e6, extent_f * 1e6)
        plt.ylabel("Intensity (arb.)")
        plt.title(f"Cross-section   $z_0={zf*100:.2f}$ cm")
        plt.tight_layout()
        plt.show()

    return xf, profile


def crosssection_xz(
    params: dict,
    extent_z: float = 20e-6,
    n_z: int = 51,
    show_focus: bool = True,
    **kwargs,
):
    """
    Compute and optionally plot |E|² at X_f≈0 as a function of y_f and z_f.

    Parameters
    ----------
    params    : dict of beam/lens parameters
    extent_z  : scan range in z (±extent_z)
    n_z       : number of z points
    alpha     : HG-mode parameter
    mmax      : highest Hermite–Gaussian index included
    show_focus : whether to display the plot

    Returns
    -------
    z_scan    : 1-D array of z_f values (meters)
    xf        : 1-D array of x_f values (meters)
    profiles  : 2-D array, shape (len(xf), len(z_scan)), |E|²(x_f, z_f)
    """
    z_scan = np.linspace(-extent_z, extent_z, n_z)
    profiles = []
    for z in tqdm(z_scan, desc="z scan"):
        xf, prof = crosssection_x(
            params, zf=z, show_focus=False, tqdm_enable=False, **kwargs
        )
        profiles.append(prof)

    profiles = np.array(profiles)  # shape (n_z, n_xf)
    profiles = profiles.T  # shape (n_xf, n_z)

    if show_focus:
        extent_f = params["extent_f"]
        extent = [
            -extent_z * 1e6,
            extent_z * 1e6,
            -extent_f * 1e6,
            extent_f * 1e6,
        ]
        plt.figure(figsize=(6, 5))
        plt.imshow(
            profiles,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="rainbow",
            vmin=0,
        )
        plt.colorbar(label="Normalised intensity")
        plt.xlabel(r"$z_0$ (µm)")
        plt.ylabel(r"$y_f$ (µm)")
        plt.title(r"Cross-section $x_f\!\approx\!0$ in the $y$–$z$ plane")
        plt.tight_layout()
        plt.show()

    return z_scan, xf, profiles


def linear_increasing_tilt(ix, iy, nx, ny, Xi, Yi):
    return (ix * (2 * np.pi / 780e-9) * 2e-5) * Xi, np.ones_like(Xi)


def linear_increasing_tilt_disc(ix, iy, nx, ny, Xi, Yi):
    return (ix * (2 * np.pi / 780e-9) * 2e-5) * (ix * 1e-3), np.ones_like(Xi)


def linear_increasing_tilt_cont(ix, iy, nx, ny, Xi, Yi):
    return ((Xi / 1e-3) * (2 * np.pi / 780e-9) * 2e-5) * Xi, np.ones_like(Xi)


def linear_increasing_tilt_only(ix, iy, nx, ny, Xi, Yi):
    return (ix * (2 * np.pi / 780e-9) * 2e-5) * (Xi - ix * 1e-3), np.ones_like(Xi)


def kxy_disc(ix, iy, nx, ny, Xi, Yi):
    return 0.3 * ix * iy, np.ones_like(Xi)


def kxy_cont(ix, iy, nx, ny, Xi, Yi):
    return 0.3 * (Xi / 1e-3) * (Yi / 1e-3), np.ones_like(Xi)


def kxpby(ix, iy, nx, ny, Xi, Yi):
    return 0.3 * ix + 0.8 * iy, np.ones_like(Xi)


DSP = -10e-6  # 20 um


def tilt_phase(tx, ty, Xi, Yi):
    k = 2 * np.pi / 780e-9
    return (tx * k) * Xi + (ty * k) * Yi, np.ones_like(Xi)


def misaligned_tilt(ix, iy, nx, ny, Xi, Yi):
    tx = -np.cos(2 * np.pi * nx / 4) * (DSP / 46.7e-3)
    return (tx * (2 * np.pi / 780e-9)) * Xi, np.ones_like(Xi)


def misaligned_displacement(ix, iy, nx, ny):
    dx = -np.sin(2 * np.pi * nx / 4) * DSP
    dy = 0
    return dx, dy


def pinhole_demo():
    NP = 5
    data_pinhole = np.zeros((2 * NP + 1, 2 * NP + 1), dtype=np.float64)
    for i in range(-NP, NP + 1):
        for j in range(-NP, NP + 1):
            center_x = i * 5e-6
            center_y = j * 5e-6
            params.update({"pinhole": {"diameter": 5e-6, "x": center_x, "y": center_y}})
            xf, yf, E_tilde, intensity = crosssection_xy(
                params, zf=0.0e-3, show_focus=False
            )
            # get the intensity sum
            sum_intensity = np.sum(intensity)

            print(
                f"Pinhole offset (x,y)=({i*5},{j*5}) um: Center intensity = {sum_intensity:.6e}"
            )
            data_pinhole[i + NP, j + NP] = sum_intensity

    plt.imshow(
        data_pinhole,
        extent=[-NP * 5, NP * 5, -NP * 5, NP * 5],
        origin="lower",
        cmap="Blues",
        aspect="equal",
        # vmin=0,
    )
    plt.show()


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
        _, _, E_tilde_0, intensity = crosssection_xy(
            params, zf=0e-6, show_E_field=True, show_focus=True
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

    elif TYPE == 4:
        pinhole_demo()
    #
    elif TYPE == 5:
        xf, yf, E_tilde_0, intensity = crosssection_xy(params, zf=0, show_focus=False)
        X, Y = np.meshgrid(xf, yf, indexing="xy")
        print(np.min(X) * 1e6, np.max(X) * 1e6, np.min(Y) * 1e6, np.max(Y) * 1e6)
        print(X[0])
        exit(0)

        tList = np.linspace(0.1e-6, 0.6e-6, 5)
        z0 = 7.8e-3
        # zList = [0]
        zList = np.arange(-2e-3, 2e-3, 0.1e-3)
        for zf in zList:
            z = z0 + zf
            gif_data = []
            for t in tList:
                print(f"Calculating for t={t*1e6:.2f} us")
                # sequences = zigzag_sequences()
                # sequences = long_transport_sequences()
                sequences = lensing_sequences()
                models = eom_model(params, t=t, sequences=sequences)
                E_tilde_sum = np.zeros_like(E_tilde_0)
                for model in models:
                    params.update({"phase_amp_func": model})
                    _, _, E_tilde_i, intensity = crosssection_xy(
                        params, zf=zf, show_focus=False
                    )
                    E_tilde_sum += E_tilde_i
                intensity = np.abs(E_tilde_sum) ** 2
                gif_data.append(intensity)
            gif_data = np.array(gif_data)
            np.savez(
                f"./data/z={z*1e3:.1f}_scan_data.npz", X=X, Y=Y, Z=gif_data, t=tList
            )
            print(gif_data.shape)

        import imageio

        gif_data = (gif_data / np.max(gif_data) * 255).astype(np.uint8)
        imageio.mimwrite("./figs/vipa_eom_demo.gif", gif_data, fps=10, loop=0)
