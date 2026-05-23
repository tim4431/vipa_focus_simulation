"""Plot the VIPA xz profile from the false-2D (crosssection_xz) and the
true-3D (crosssection_xy at each z, y=y0 slice) simulations side by side.

Loads precomputed arrays — run these first:
- python example/ripa_vs_gaussian_focus.py      → data/ripa_vs_gaussian_focus.npz
- python example/ripa_vs_gaussian_focus_3d.py   → data/ripa_vs_gaussian_focus_3d.npz

Output: example/render/ripa_xz_2d_vs_3d.png (dpi=600)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt


def load_xz(npz_path):
    d = np.load(npz_path)
    xf = d["xf"]
    z_scan = d["z_scan"]
    profiles = d["profiles"]  # shape (n_xf, n_z)
    return xf, z_scan, profiles / np.max(profiles)


if __name__ == "__main__":
    CMAP = "Blues"
    X0 = 0.0
    Z0 = 0.0

    xf_2d, z_2d, img_2d = load_xz(ROOT / "data/ripa_vs_gaussian_focus.npz")
    xf_3d, z_3d, img_3d = load_xz(ROOT / "data/ripa_vs_gaussian_focus_3d.npz")

    def extent_for(xf, z_scan):
        return [
            (z_scan.min() - Z0) * 1e6,
            (z_scan.max() - Z0) * 1e6,
            (xf.min() - X0) * 1e6,
            (xf.max() - X0) * 1e6,
        ]

    fig, (ax_2d, ax_3d) = plt.subplots(
        2, 1, figsize=(6, 7), sharex=True, sharey=True
    )
    for ax, img, xf, z_scan, label in (
        (ax_2d, img_2d, xf_2d, z_2d, "false-2D (crosssection_xz)"),
        (ax_3d, img_3d, xf_3d, z_3d, "true-3D (xy@z, y=y0)"),
    ):
        im = ax.imshow(
            img,
            extent=extent_for(xf, z_scan),
            origin="lower",
            aspect="equal",
            cmap=CMAP,
            vmin=0,
            vmax=1,
        )
        ax.set_ylabel(r"$x$ (µm)")
        ax.text(
            0.02, 0.95, label,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="k",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax_3d.set_xlabel(r"$z$ (µm)")

    cbar = fig.colorbar(
        im,
        ax=[ax_2d, ax_3d],
        ticks=[0, 1],
        orientation="horizontal",
        location="bottom",
        pad=0.02,
        fraction=0.05,
        shrink=0.8,
    )
    cbar.set_label("Intensity (a.u.)")

    out_png = ROOT / "example/render/ripa_xz_2d_vs_3d.png"
    fig.savefig(out_png, dpi=600)
    print(f"✓  Saved {out_png}")
    plt.show()
