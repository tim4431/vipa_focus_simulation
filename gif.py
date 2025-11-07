# ----------------------------------------------------------------------
# Animated GIF:   phase φ  ∈ [0, 2π]  →  y–z cross-section movie
# ----------------------------------------------------------------------
from vipa_focus import *
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # <-- v2 API gives PIL Images

from io import BytesIO  # in-memory PNG for each frame

# ----- simulation parameters ------------------------------------------
params = {
    "Nn": 20,
    "w": 100e-6,
    "d": 500e-6,
    "f": 0.01,
    "phi": np.pi / 2,
    "lambda": 780e-9,
    "D": 4e-2,
    "RESOLUTION_X": 20e-6,
    "extent_f": 20e-6,
}

NPHI = 40  # frames along the φ-scan
DUR = 0.08  # seconds per frame in the GIF

# container for GIF frames
frames = []

# ----- main loop over φ ------------------------------------------------
for phi in np.linspace(0.0, 2 * np.pi, NPHI, endpoint=False):
    print(f"phi = {phi:.2f}")
    params["phi"] = phi  # update phase

    EXTENT_Z = 20e-6
    z_scan = np.linspace(-EXTENT_Z, EXTENT_Z, 101)
    profiles = []
    for z in z_scan:
        # print(f"z0 = {z:.2f} m")
        _, prof = crosssection_x(
            params,
            z0=z,
            alpha=1,
            normalize=False,
            show_plot=False,
        )
        profiles.append(prof)

    profiles = np.array(profiles)  # shape (nz, ny)
    profiles = profiles.T
    print(np.max(profiles.flatten()) / 1e16)

    # ------- plot this frame ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    EXTENT_F = params["extent_f"]
    extent = [
        -EXTENT_Z * 1e6,
        EXTENT_Z * 1e6,
        -EXTENT_F * 1e6,
        EXTENT_F * 1e6,
    ]
    im = ax.imshow(
        profiles,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="rainbow",
        vmin=0,
    )
    ax.set_xlabel(r"$z_0$ (µm)")
    ax.set_ylabel(r"$y_f$ (µm)")
    ax.set_title(rf"$x_f \approx 0$   —   $\varphi = {phi:.2f}$")
    # always set the same color limits [0,800] for all frames
    im.set_clim(0, 3.4e16)
    fig.colorbar(im, ax=ax, label="Intensity (arb.)")
    fig.tight_layout()

    # save the figure to an in-memory PNG and append as a frame
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    frames.append(imageio.imread(buf))

# ----- write the animated GIF -----------------------------------------
imageio.mimsave("scan_phi.gif", frames, duration=DUR)
print("✓  GIF saved as scan_phi.gif")
