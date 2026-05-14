"""Misalignment displacement scan (was vipa_focus.py TYPE == 4)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import imageio

import src.vipa_focus as vipa_focus
from src.vipa_focus import PARAMS_80, vipa_rays
from src.crosssections import crosssection_x


if __name__ == "__main__":
    params = PARAMS_80

    dsps = np.linspace(-200e-6, 200e-6, 11)
    gifs = []
    for dsp in dsps:
        vipa_focus.DSP = dsp
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
    imageio.mimsave(ROOT / "example/render/vipa_misalignment_dsp.gif", gifs, fps=5, loop=0)
    print("✓  GIF saved as vipa_misalignment_dsp.gif")
