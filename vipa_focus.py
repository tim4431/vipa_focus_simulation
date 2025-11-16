import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from typing import Tuple
from scipy.special import eval_hermite
from elec import *
from tqdm import tqdm
import time
from typing import List
import imageio


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _cm_coefficient(m: int, alpha: float) -> float:
    """
    Normal-ordering coefficient appearing in the HG mode expansion.
    """
    if m == 0:
        # `eval_hermite(0, 0) == 1`, but saving the call avoids
        # triggering a SciPy warning when alpha==1, mmax<=1.
        Hm0 = 1.0
    else:
        Hm0 = eval_hermite(m, 0)

    return Hm0 * np.sqrt(
        (2 * alpha * (alpha**2 - 1) ** m)
        / (2**m * factorial(m) * (1 + alpha**2) ** (m + 1))
    )


def vipa_elec2d(
    Xi: np.ndarray,
    Yi: np.ndarray,
    params: dict,
    tqdm_enable: bool = True,
) -> np.ndarray:
    """
    Complex electric field in the source plane.

    Parameters
    ----------
    Xi, Yi : 2-D ndarrays returned by `np.meshgrid`
    params : dict containing Nn, w, d, phi, lambda
    zf    : axial offset from the waist
    alpha : HG-mode parameter (alpha == 1 ⇒ pure Gaussian)
    mmax  : highest Hermite–Gaussian index included
    """
    # Unpack parameters with shorter names
    Nx = params["Nx"]
    Ny = params["Ny"]
    lx = params["lx"]
    ly = params["ly"]
    FSR_Ratio = params["FSR_Ratio"]
    w = params["w"]
    d = params["d"]
    phi = params["phi"]
    wl = params["lambda"]
    alpha = params["alpha"] if "alpha" in params else 1.0
    mmax = params["mmax"] if "mmax" in params else 1
    #
    phase_amp_func = (
        params["phase_amp_func"]
        if "phase_amp_func" in params
        else lambda ix, iy, Xi, Yi: (0.0, 0, 0)
    )

    field = np.zeros_like(Xi, dtype=np.complex128)

    # if alpha == 1 or mmax <= 1:  # pure Gaussian beams
    #     for n in range(n_start, n_end + 1):
    #         field += np.exp(
    #             -1j * n * phi - ((Xi - n * d) ** 2 + Yi**2) / (w**2) - 0.1 * (n - n_start)
    #         )
    #     field /= w**2
    #     return field

    # More general case with HG modes
    if tqdm_enable:
        iterator = tqdm(range(Nx * Ny), desc="(nx, ny) loop")
    else:
        iterator = range(Nx * Ny)
    #
    # Find 1D indices where Xi and Yi are within cutoff
    Xi_1d = Xi[0, :]
    Yi_1d = Yi[:, 0]
    for idx in iterator:
        nx = idx // Ny
        ny = idx % Ny
        #
        ix = nx - (Nx - 1) / 2
        iy = ny - (Ny - 1) / 2

        # Only calculate damping within 10 sigma (square mask)
        sigma = w / np.sqrt(2)  # standard deviation
        cutoff = 10 * sigma

        # Deduce row and column ranges from meshgrid
        center_x = ix * d
        center_y = iy * d

        c1 = np.searchsorted(Xi_1d, center_x - cutoff)
        c2 = np.searchsorted(Xi_1d, center_x + cutoff)
        r1 = np.searchsorted(Yi_1d, center_y - cutoff)
        r2 = np.searchsorted(Yi_1d, center_y + cutoff)

        # Clamp to array bounds
        c1 = max(0, c1)
        c2 = min(Xi.shape[1], c2 + 1)
        r1 = max(0, r1)
        r2 = min(Xi.shape[0], r2 + 1)

        # damping = np.zeros_like(Xi, dtype=np.complex128)
        if c2 > c1 and r2 > r1:
            Xi_slice = Xi[r1:r2, c1:c2]
            Yi_slice = Yi[r1:r2, c1:c2]

            damping_patched = np.exp(
                -((Xi_slice - ix * d) ** 2 + (Yi_slice - iy * d) ** 2) / (w**2)
                - lx * nx
                - ly * ny
            )

        if params.get("phase_amp_func"):
            phase_res, amp_res = phase_amp_func(ix, iy, Xi_slice, Yi_slice)
            phase_additional_patched = np.exp(1j * phase_res) * amp_res
        else:
            phase_additional_patched = np.ones_like(Xi_slice)

        if alpha == 1 or mmax < 1:
            field[r1:r2, c1:c2] += (
                damping_patched
                * phase_additional_patched
                * np.exp(-1j * (ix * FSR_Ratio + iy) * (phi))
                / w**2
            )
        else:
            for p in range(mmax + 1):
                for q in range(mmax + 1):
                    coeff = _cm_coefficient(p, alpha) * _cm_coefficient(q, alpha) / w**2
                    phase = np.exp(
                        -1j
                        * (ix * FSR_Ratio + iy)
                        * (phi + (p + q) * 2 * np.arctan(2.0))
                    )
                    # Hp = eval_hermite(p, np.sqrt(2) * Xi / w)
                    Hp = eval_hermite(p, np.sqrt(2) * Xi_slice / w)
                    # Hq = eval_hermite(q, np.sqrt(2) * Yi / w)
                    Hq = eval_hermite(q, np.sqrt(2) * Yi_slice / w)
                    #
                    field[r1:r2, c1:c2] += (
                        coeff
                        * phase
                        * damping_patched
                        * Hp
                        * Hq
                        * phase_additional_patched
                    )

    return field


# ──────────────────────────────────────────────────────────────────────
# convenience wrapper – square ±extent_f patch
# ──────────────────────────────────────────────────────────────────────
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
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    # print(Xi.shape, Yi.shape)
    Ei = vipa_elec2d(Xi, Yi, params, **kwargs)
    zfi = params.get("zfi", None)
    if zfi is not None:
        print(f"Propagating initial zf = {zfi*1e3:.2f} mm")
        Ei = freespace_propagation(xi, yi, Ei, params["lambda"], zfi)
    if show_E_field:
        extent = [-D / 2 * 1e6, D / 2 * 1e6, -D / 2 * 1e6, D / 2 * 1e6]
        plt.figure(figsize=(6, 5))
        plt.imshow(
            np.abs(Ei) ** 2,
            extent=extent,
            origin="lower",
            cmap="rainbow",
            aspect="equal",
            vmin=0,
        )
        plt.colorbar(label="Intensity (arb.)")
        plt.xlabel(r"$x$ (source plane, $\mu$m)")
        plt.ylabel(r"$y$ (source plane, $\mu$m)")
        plt.title("Source plane intensity")
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

    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    Ei = vipa_elec2d(Xi, Yi, params, **kwargs)
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


def linear_increasing_tilt(ix, iy, Xi, Yi):
    return (ix * (2 * np.pi / 780e-9) * 2e-5) * Xi, np.ones_like(Xi)


def linear_increasing_tilt_disc(ix, iy, Xi, Yi):
    return (ix * (2 * np.pi / 780e-9) * 2e-5) * (ix * 1e-3), np.ones_like(Xi)


def linear_increasing_tilt_cont(ix, iy, Xi, Yi):
    return ((Xi / 1e-3) * (2 * np.pi / 780e-9) * 2e-5) * Xi, np.ones_like(Xi)


def linear_increasing_tilt_only(ix, iy, Xi, Yi):
    return (ix * (2 * np.pi / 780e-9) * 2e-5) * (Xi - ix * 1e-3), np.ones_like(Xi)


def kxy_disc(ix, iy, Xi, Yi):
    return 0.3 * ix * iy, np.ones_like(Xi)


def kxy_cont(ix, iy, Xi, Yi):
    return 0.3 * (Xi / 1e-3) * (Yi / 1e-3), np.ones_like(Xi)


def kxpby(ix, iy, Xi, Yi):
    return 0.3 * ix + 0.8 * iy, np.ones_like(Xi)


def misaligned_tilt(ix, iy, Xi, Yi):
    tx = np.cos(2 * np.pi * ix / 4) * 2e-5
    return (tx * (2 * np.pi / 780e-9)) * Xi, np.ones_like(Xi)


def phase_amp_eom(params, t, ix, iy, Xi, Yi, phase_func, freq_func, amp_func):
    c = 3e8
    FSR_Ratio = params["FSR_Ratio"]
    Lrt = params["Lrt"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    t_travel = ((ix + (Nx - 1) / 2) + ((iy + (Ny - 1) / 2) / FSR_Ratio)) * Lrt / c
    t_eom = t - t_travel
    phase = phase_func(t_eom)
    amp = amp_func(t_eom)

    return (
        phase * np.ones_like(Xi),
        amp * np.ones_like(Xi),
    )


def phase_freq_func_from_sequence(t_arr, frequency_arr, amp_arr):
    """
    t_arr=[(t0,t1),(t1,t2),...]
    frequency_arr=[(f0,f1), (f1,f2), ...] frequency ramps
    amp_arr=[a0,a1,...] amplitude levels
    keep a phase_end object to store the phase at the end of each segment, and keep phase continuity
    """
    assert len(t_arr) == len(
        frequency_arr
    ), "t_arr and frequency_arr must have the same length"
    freq_func = lambda t: np.piecewise(
        t,
        [t < t_arr[0][0]]
        + [(t >= t_arr[i][0]) & (t < t_arr[i][1]) for i in range(len(t_arr))]
        + [t >= t_arr[-1][1]],
        [frequency_arr[0][0]]
        + [
            lambda t, i=i: frequency_arr[i][0]
            + (frequency_arr[i][1] - frequency_arr[i][0])
            * (t - t_arr[i][0])
            / (t_arr[i][1] - t_arr[i][0])
            for i in range(len(t_arr))
        ]
        + [frequency_arr[-1][1]],
    )

    #
    phase_start = np.zeros(len(t_arr) + 1)
    for i in range(len(t_arr)):
        dt = t_arr[i][1] - t_arr[i][0]
        f0 = frequency_arr[i][0]
        f1 = frequency_arr[i][1]
        phase_start[i + 1] = phase_start[i] + 2 * np.pi * (f0 + f1) / 2 * dt
    # print(frequency_arr)
    phase_func = lambda t: np.piecewise(
        t,
        [t < t_arr[0][0]]
        + [(t >= t_arr[i][0]) & (t < t_arr[i][1]) for i in range(len(t_arr))]
        + [t >= t_arr[-1][1]],
        [0.0]
        + [
            lambda t, i=i: phase_start[i]
            + 2 * np.pi * frequency_arr[i][0] * (t - t_arr[i][0])
            + np.pi
            * (frequency_arr[i][1] - frequency_arr[i][0])
            * (t - t_arr[i][0]) ** 2
            / (t_arr[i][1] - t_arr[i][0])
            for i in range(len(t_arr))
        ]
        + [phase_start[-1]],
    )
    #
    amp_func = lambda t: np.piecewise(
        t,
        [t < t_arr[0][0]]
        + [(t >= t_arr[i][0]) & (t < t_arr[i][1]) for i in range(len(t_arr))]
        + [t >= t_arr[-1][1]],
        [0.0] + [lambda t, i=i: amp_arr[i] for i in range(len(t_arr))] + [0.0],
    )
    return phase_func, freq_func, amp_func


def eom_model(params, t, sequences: List):

    models = []
    for seq in sequences:
        t_arr = seq["t_arr"]
        frequency_arr = seq["frequency_arr"]
        amp_arr = seq["amp_arr"]
        #
        phase_func, freq_func, amp_func = phase_freq_func_from_sequence(
            t_arr, frequency_arr, amp_arr
        )
        phase_amp_eom_i = lambda ix, iy, Xi, Yi: phase_amp_eom(
            params,
            t,
            ix,
            iy,
            Xi,
            Yi,
            phase_func=phase_func,
            freq_func=freq_func,
            amp_func=amp_func,
        )
        models.append(phase_amp_eom_i)

    return models


def zigzag_sequences():
    T = 0.2e-6
    FSR2 = 0.1298e9
    F0 = 11.0e9
    F1 = F0 + 0.1 * FSR2
    F2 = F0 + 0.3 * FSR2
    F3 = F0 + FSR2 + 0.5 * FSR2
    F4 = F0 + FSR2 + 0.7 * FSR2
    # seq0
    t_arr_0 = [(0, 2 * T), (2 * T, 4 * T)]
    frequency_arr_0 = [(F2, F1), (F1, F2)]
    amp_arr_0 = [1.0, 1.0]
    seq0 = {
        "t_arr": t_arr_0,
        "frequency_arr": frequency_arr_0,
        "amp_arr": amp_arr_0,
    }
    # seq1
    t_arr_1 = [(0, T), (T, 2 * T), (2 * T, 3 * T), (3 * T, 4 * T)]
    frequency_arr_1 = [(0, 0), (F4, F4), (F4, F3), (F3, F4)]
    amp_arr_1 = [0.0, 1.0, 1.0, 1.0]
    seq1 = {
        "t_arr": t_arr_1,
        "frequency_arr": frequency_arr_1,
        "amp_arr": amp_arr_1,
    }
    return [seq0, seq1]


def long_transport_sequences():
    T = 1e-6
    FSR2 = 0.1298e9
    F0 = 11.0e9
    F1 = F0 + 4 * FSR2
    # seq0
    t_arr_0 = [(0, T)]
    frequency_arr_0 = [(F0, F1)]
    amp_arr_0 = [1.0]
    seq0 = {
        "t_arr": t_arr_0,
        "frequency_arr": frequency_arr_0,
        "amp_arr": amp_arr_0,
    }
    return [seq0]


def lensing_sequences():
    T = 0.2e-6
    FSR2 = 0.1298e9
    F0 = 11.0e9
    F1 = F0 + 0.5 * FSR2
    # seq0
    t_arr_0 = [(0, T), (T, 2 * T), (2 * T, 3 * T)]
    frequency_arr_0 = [(F0, F0), (F0, F1), (F1, F1)]
    amp_arr_0 = [1.0, 1.0, 1.0]
    seq0 = {
        "t_arr": t_arr_0,
        "frequency_arr": frequency_arr_0,
        "amp_arr": amp_arr_0,
    }
    return [seq0]


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
    params_100 = {
        "Nx": 100,
        "Ny": 100,
        "FSR_Ratio": 100,
        "lx": 0.02,
        "ly": 0.02,
        "w": 108e-6,  # beam waist
        "d": 300e-6,  # beam spacing
        "f": 0.2,
        "phi": 0.0,
        "lambda": 780e-9,
        "D": 20e-2,  # real space extent
        "RESOLUTION_X": 25e-6,  # real space resolution
        "extent_f": 300e-6,  # focal plane extent, only for plotting
    }
    params = {
        # "Nx": 8,
        "Nx": 1,
        # "Ny": 9,
        "Ny": 1,
        "FSR_Ratio": 22.0,
        "Lrt": 2.311,
        "lx": 0.15,
        "ly": 0.052,
        "w": 108e-6,  # beam waist
        "d": 1000e-6,  # beam spacing
        "f": 0.2,
        "phi": 0.0,
        "lambda": 780e-9,
        "D": 10e-2,  # real space extent
        # "D": 4e-2,  # real space extent
        "RESOLUTION_X": 25e-6,  # real space resolution
        # "extent_f": 40e-6,  # focal plane extent, only for plotting
        "extent_f": 1500e-6,  # focal plane extent, only for plotting
        # "extent_f": 100e-6,  # focal plane extent, only for plotting
        # "extent_f": 800e-6,  # focal plane extent, only for plotting
        # "alpha": 2.0,
        # "phase_amp_func": misaligned_tilt,
        # "phase_amp_func": None,
        "zfi": None,
        # "pinhole": {"diameter": 5e-6, "x": 0.0, "y": 0.0, "zf_pinhole": 0.5e-3},
    }

    TYPE = 0

    if TYPE == 0:
        phi = 0
        params.update({"phi": phi})
        zf = 0
        # _, _, _, intensity = crosssection_xy(params, zf=zf)
        # H, W = intensity.shape
        _, _, E_tilde_0, intensity = crosssection_xy(
            params, zf=0e-6, show_E_field=False
        )
        # phi = 0.62
        # params.update({"phi": phi})
        # _, _, E_tilde_1, intensity = crosssection_xy(
        #     params, zf=0e-6, show_E_field=False
        # )
        # E_tilde_sum = E_tilde_0 + E_tilde_1
        # intensity = np.abs(E_tilde_sum) ** 2
        # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # ax.set_axis_off()
        # im = ax.imshow(
        #     intensity,
        #     cmap="Reds",
        # )
        # plt.tight_layout()
        # plt.savefig(
        #     f"./figs/vipa_focus_demo_intensity_2d.png",
        #     dpi=600,
        #     bbox_inches="tight",
        #     transparent=True,
        # )

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
