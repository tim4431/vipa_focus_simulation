import numpy as np
from typing import Tuple, List


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
