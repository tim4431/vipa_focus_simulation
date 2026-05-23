"""Vectorial XY focal plane profile for a VIPA field and high-NA objective."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.high_na_polarization import (
    FOV,
    NA,
    N_MEDIUM,
    POLARIZATION,
    component_energy_fractions,
    format_percent,
    plot_component_rows,
    simulate_vipa_vector_xy,
)
from src.vipa_focus import PARAMS_10_TWZ


if __name__ == "__main__":
    params = PARAMS_10_TWZ
    zf = 0.0

    xf, yf, E_focus, obs, used_params = simulate_vipa_vector_xy(
        params,
        objective_na=NA,
        n=N_MEDIUM,
        polarization=POLARIZATION,
        fov=FOV,
        zf=zf,
    )

    FSR_L = params["lambda"] * params["f"] / params["d"]
    fractions = component_energy_fractions(obs)
    print("FSR_L =", FSR_L)
    print(f"source grid D = {used_params['D'] * 1e3:.2f} mm")
    print(f"source grid dx = {used_params['RESOLUTION_X'] * 1e6:.1f} um")
    print(
        "source grid shape = "
        f"{int(used_params['D'] / (2 * used_params['RESOLUTION_X'])) * 2 + 1}"
    )
    print(f"focal pixel size = {(xf[1] - xf[0]) * 1e9:.1f} nm")
    print(f"focus array shape = {E_focus.shape}")
    print(f"FOV energy fraction y = {format_percent(fractions['y'])}")
    print(f"FOV energy fraction z = {format_percent(fractions['z'])}")

    plot_component_rows(
        xf,
        yf,
        obs,
        rf"VIPA vector focus, zf={zf * 1e6:.1f} um",
        fov=FOV,
    )
