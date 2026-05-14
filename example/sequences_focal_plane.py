"""EOM time-sequence focal-plane animation (was sequences.py TYPE == 0)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import imageio

from src.vipa_focus import PARAMS_10, vipa_rays
from src.crosssections import crosssection_xy
from src.sequences import (
    eom_model,
    zigzag_sequences,
    long_transport_sequences,
    lensing_sequences,
    pulse_sequences,
)


if __name__ == "__main__":
    params = PARAMS_10

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
    np.savez(ROOT / f"data/sequences_1.npz", X=X, Y=Y, Z=gif_data, t=tList)
    print(gif_data.shape)

    gif_data = (gif_data / np.max(gif_data) * 255).astype(np.uint8)
    imageio.mimwrite(ROOT / "example/render/vipa_eom_demo.gif", gif_data, fps=10, loop=0)
