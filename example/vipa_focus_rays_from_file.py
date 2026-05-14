"""Focal plane intensity from optical-design traced rays (was vipa_focus.py TYPE == 5)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vipa_focus import PARAMS_80, rays_from_file
from src.crosssections import crosssection_xy


if __name__ == "__main__":
    params = PARAMS_80

    rays = rays_from_file(ROOT / "data/optable/ripa_gen2_2nd_mon0_rays.npz", params)
    xf, yf, E_tilde_0, intensity = crosssection_xy(
        rays, params, zf=0e-6, show_E_field=False, show_focus=True, log_scale=False
    )
