"""EOM time-sequence animations.

MODE = "focal_plane" -> XY focal-plane animation (was sequences.py TYPE == 0)
MODE = "xz_lensing"  -> XZ lensing-sequence animation (was sequences.py TYPE == 1)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import imageio

from src.vipa_focus import PARAMS_10, vipa_rays
from src.crosssections import crosssection_xy, crosssection_xz_naive
from src.sequences import (
    eom_model,
    zigzag_sequences,
    long_transport_sequences,
    lensing_sequences,
    pulse_sequences,
)

MODE = "focal_plane"  # "focal_plane" or "xz_lensing"


def run_focal_plane():
    params = PARAMS_10

    # sequences = zigzag_sequences()
    # sequences = long_transport_sequences()
    # sequences = lensing_sequences()
    sequences = pulse_sequences()

    models = eom_model(params, t=1e-6, sequences=sequences)
    model = models[0]
    params.update({"phase_amp_func": model})
    rays = vipa_rays(params)
    xf, yf, E_tilde_0, intensity_0 = crosssection_xy(rays, params, show_focus=True)
    X, Y = np.meshgrid(xf, yf, indexing="xy")

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
    np.savez(ROOT / "data/sequences_1.npz", X=X, Y=Y, Z=gif_data, t=tList)
    print(gif_data.shape)

    gif_data = (gif_data / np.max(gif_data) * 255).astype(np.uint8)
    imageio.mimwrite(
        ROOT / "example/render/vipa_eom_demo.gif", gif_data, fps=10, loop=0
    )


def run_xz_lensing():
    params = PARAMS_10

    T = 2e-6
    sequences = lensing_sequences(T=T)
    params.update({"extent_f": 80e-6})

    models = eom_model(params, t=T / 2, sequences=sequences)
    model = models[0]
    params.update({"phase_amp_func": model})
    rays = vipa_rays(params)
    EXTENT_Z = 12e-3
    NZ = 300
    z_scan, xf, profiles = crosssection_xz_naive(
        rays, params, extent_z=EXTENT_Z, n_z=NZ, show_focus=True
    )
    Z, X = np.meshgrid(z_scan, xf, indexing="xy")

    tList = np.linspace(0.0e-6, 2 * T, 200)
    gif_data = []
    for t in tList:
        print(f"Calculating for t={t*1e6:.2f} us")
        models = eom_model(params, t=t, sequences=sequences)
        E_tilde_sum = np.zeros_like(profiles)
        for model in models:
            params.update({"phase_amp_func": model})
            rays = vipa_rays(params)
            z_scan, xf, profiles = crosssection_xz_naive(
                rays, params, extent_z=EXTENT_Z, n_z=NZ, show_focus=False
            )
            E_tilde_sum += profiles
        intensity = np.abs(E_tilde_sum) ** 2
        gif_data.append(intensity)
    gif_data = np.array(gif_data)
    np.savez(
        ROOT / f"data/sequences_lensing_T={T*1e9:.1f}.npz",
        X=X,
        Z=Z,
        I=gif_data,
        t=tList,
    )
    print(gif_data.shape)

    xmList, zmList, IList, xMList, zMList = [], [], [], [], []
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
        ROOT / f"data/sequences_lensing_T={T*1e9:.1f}_mean_position.npz",
        xm=xmList,
        zm=zmList,
        t=tList,
        I=IList,
        xM=xMList,
        zM=zMList,
    )

    gif_data = (gif_data / np.max(gif_data) * 255).astype(np.uint8)
    imageio.mimwrite(
        ROOT / f"example/render/vipa_eom_lensing_T={T*1e9:.1f}.gif",
        gif_data,
        fps=20,
        loop=0,
    )


if __name__ == "__main__":
    if MODE == "focal_plane":
        run_focal_plane()
    elif MODE == "xz_lensing":
        run_xz_lensing()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")
