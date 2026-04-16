import numpy as np
from typing import Tuple, List
from crosssections import *
from vipa_focus import *


def phase_amp_eom(params, t, ix, iy, nx, ny, Xi, Yi, phase_func, freq_func, amp_func):
    c = 3e8
    FSR_Ratio = params["FSR_Ratio"]
    Lrt = params["Lrt"]
    t_travel = (nx + ny / FSR_Ratio) * Lrt / c
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
        phase_amp_eom_i = lambda ix, iy, nx, ny, Xi, Yi: phase_amp_eom(
            params,
            t,
            ix,
            iy,
            nx,
            ny,
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


def lensing_sequences(T: float = 0.1e-6):
    # T = 0.05e-6
    FSR2 = 0.1298e9
    F0 = 11.0e9
    F1 = F0 + 0.5 * FSR2
    # seq0
    t_arr_0 = [(0, T / 2), (T / 2, 3 * T / 2), (3 * T / 2, 2 * T)]
    # t_arr_0 = [(0, T), (T, 2 * T)]
    frequency_arr_0 = [(F0, F0), (F0, F1), (F1, F1)]
    # frequency_arr_0 = [(F0, F0), (F1, F1)]
    amp_arr_0 = [1.0, 1.0, 1.0]
    # amp_arr_0 = [1.0, 1.0]
    seq0 = {
        "t_arr": t_arr_0,
        "frequency_arr": frequency_arr_0,
        "amp_arr": amp_arr_0,
    }
    return [seq0]


def pulse_sequences():
    T = 1e-6
    F0 = 11.295e9
    # seq0
    t_arr_0 = [(0, T / 2), (T / 2, 3 * T / 2), (3 * T / 2, 2 * T)]
    frequency_arr_0 = [(0, 0), (F0, F0), (0, 0)]
    amp_arr_0 = [0.0, 1.0, 0.0]
    seq0 = {
        "t_arr": t_arr_0,
        "frequency_arr": frequency_arr_0,
        "amp_arr": amp_arr_0,
    }
    return [seq0]


if __name__ == "__main__":
    params = PARAMS_10

    TYPE = 2

    if TYPE == 0:
        # sequences = zigzag_sequences()
        # sequences = long_transport_sequences()
        # sequences = lensing_sequences()
        sequences = pulse_sequences()
        #
        models = eom_model(params, t=1e-6, sequences=sequences)
        model = models[0]
        params.update({"phase_amp_func": model})
        rays = vipa_rays(params)
        xf, yf, E_tilde_0, intensity_0 = crosssection_xy(rays, params, show_focus=True)
        X, Y = np.meshgrid(xf, yf, indexing="xy")
        # exit(0)
        #
        tList = np.linspace(0.20e-6, 0.9e-6, 800)
        gif_data = []
        for t in tList:
            print(f"Calculating for t={t*1e6:.2f} us")
            models = eom_model(params, t=t, sequences=sequences)
            E_tilde_sum = np.zeros_like(E_tilde_0)
            for model in models:
                params.update({"phase_amp_func": model})
                rays = vipa_rays(params)
                xf, yf, E_tilde_i, intensity = crosssection_xy(
                    rays, params, show_focus=False
                )
                E_tilde_sum += E_tilde_i
            intensity = np.abs(E_tilde_sum) ** 2
            gif_data.append(intensity)
        gif_data = np.array(gif_data)
        np.savez(f"./data/sequences_1.npz", X=X, Y=Y, Z=gif_data, t=tList)
        print(gif_data.shape)

        import imageio

        gif_data = (gif_data / np.max(gif_data) * 255).astype(np.uint8)
        imageio.mimwrite("./figs/vipa_eom_demo.gif", gif_data, fps=10, loop=0)

    elif TYPE == 1:
        # sequences = zigzag_sequences()
        # sequences = long_transport_sequences()
        # sequences = lensing_sequences()
        T = 2e-6
        sequences = lensing_sequences(T=T)
        params.update({"extent_f": 80e-6})
        #
        models = eom_model(params, t=T / 2, sequences=sequences)
        model = models[0]
        params.update({"phase_amp_func": model})
        rays = vipa_rays(params)
        EXTENT_Z = 12e-3
        NZ = 300
        z_scan, xf, profiles = crosssection_xz(
            rays, params, extent_z=EXTENT_Z, n_z=NZ, show_focus=True
        )
        # xf, yf, E_tilde_0, intensity_0 = crosssection_xy(rays, params, show_focus=True)
        Z, X = np.meshgrid(z_scan, xf, indexing="xy")
        # exit(0)
        #
        tList = np.linspace(0.0e-6, 2 * T, 200)
        gif_data = []
        for t in tList:
            print(f"Calculating for t={t*1e6:.2f} us")
            models = eom_model(params, t=t, sequences=sequences)
            E_tilde_sum = np.zeros_like(profiles)
            for model in models:
                params.update({"phase_amp_func": model})
                rays = vipa_rays(params)
                z_scan, xf, profiles = crosssection_xz(
                    rays, params, extent_z=EXTENT_Z, n_z=NZ, show_focus=False
                )
                E_tilde_sum += profiles
            intensity = np.abs(E_tilde_sum) ** 2
            gif_data.append(intensity)
        gif_data = np.array(gif_data)
        np.savez(
            f"./data/sequences_lensing_T={T*1e9:.1f}.npz", X=X, Z=Z, I=gif_data, t=tList
        )
        print(gif_data.shape)
        # for every t, calculate the intensity-weighted x and z mean position
        xmList = []
        zmList = []
        IList = []
        xMList = []
        zMList = []
        for i in range(gif_data.shape[0]):
            I = gif_data[i, :, :]
            I_sum = np.sum(np.sum(I))
            if I_sum == 0:
                x_mean = 0.0
                z_mean = 0.0
                xM = 0.0
                zM = 0.0
            else:
                x_mean = np.sum(X * I) / I_sum
                z_mean = np.sum(Z * I) / I_sum
                idx = np.unravel_index(np.argmax(I), I.shape)
                zM = Z[idx]
                xM = X[idx]

            xmList.append(x_mean)
            zmList.append(z_mean)
            IList.append(I_sum)
            xMList.append(xM)
            zMList.append(zM)

        xmList = np.array(xmList)
        zmList = np.array(zmList)
        IList = np.array(IList)
        xMList = np.array(xMList)
        zMList = np.array(zMList)
        np.savez(
            f"./data/sequences_lensing_T={T*1e9:.1f}_mean_position.npz",
            xm=xmList,
            zm=zmList,
            t=tList,
            I=IList,
            xM=xMList,
            zM=zMList,
        )

        import imageio

        gif_data = (gif_data / np.max(gif_data) * 255).astype(np.uint8)
        imageio.mimwrite(
            f"./figs/vipa_eom_lensing_T={T*1e9:.1f}.gif", gif_data, fps=20, loop=0
        )
