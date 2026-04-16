# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Numerical simulation of the focal pattern produced by a VIPA (Virtually Imaged Phased Array) using Fourier optics. The code models a 2D grid of Gaussian beams (rays) emitted in a source plane, optionally modulated by an EOM phase/amplitude sequence, then propagated through an ideal lens to a focal plane.

## Running

There is no build system, test suite, or package manifest. Each top-level script is both a library module and a demo driver with an `if __name__ == "__main__":` block that dispatches on a local `TYPE = N` integer. To run a specific demo, edit the `TYPE` value at the bottom of the file and run the script:

```bash
python vipa_focus.py    # VIPA focal-plane patterns; TYPE selects xy, xz, phi-scan, misalignment, or optable rays
python sequences.py     # EOM time-sequence simulations (transport, lensing, pulses); writes .npz + .gif
python sensor.py        # Convolves stored field sequences with a photodetector transfer function
python fit_gaussian.py  # Self-test of the 1D/2D Gaussian fitting utilities
```

Dependencies (inferred from imports; install with pip): `numpy`, `scipy`, `matplotlib`, `tqdm`, `imageio`.

Outputs: `.npz` / `.npy` arrays go to [data/](data/), animated GIFs go to `figs/` (directory may need to be created).

## Architecture

The pipeline is one-way: **rays -> source-plane field -> lens FFT -> focal-plane field -> cross-section**.

1. **Ray generation** ([vipa_focus.py](vipa_focus.py)): `vipa_rays(params)` produces a `List[dict]` describing a 2D grid of Gaussian beamlets. Each ray dict carries `x, y, w` (center/waist), `ix, iy, nx, ny` (grid indices), `intensity`, `phase`, and optionally a `phase_amp_func(Xi, Yi) -> (phase, amp)` callable for per-ray spatial modulation. `rays_from_file` loads traced rays from an optical-design export (units in cm, converted to m).

2. **Source-plane assembly** ([core.py:28](core.py#L28)): `rays2elec2d(Xi, Yi, rays, params)` sums Gaussian beamlets onto a 2D complex field grid. Each beamlet is patched onto a `±5w` sub-window around its center for efficiency. When `alpha != 1` and `mmax >= 1`, the beam is expanded in a Hermite-Gaussian basis with a confocal Gouy phase `(p+q+1) * gouy_phase` per `(ix, iy)` index; otherwise a pure Gaussian with `(ix+iy) * gouy_phase` is used. `_cm_coefficient` provides the HG expansion coefficients.

3. **Lens propagation** ([core.py:118](core.py#L118)): `calc_field_after_lens(xi, yi, Ei, wl, f, zf)` implements a single FFT mapping from the input (source) plane to the focal plane. `xf, yf` are determined by the input grid spacing via `xf = wl*f*kxi/(2*pi)`. An optional `zf` offset applies an angular-spectrum phase `exp(j*kzf*zf)` *before* the FFT (near-field propagation in the lens's back focal space).

4. **Free-space propagation** ([core.py:261](core.py#L261)): `freespace_propagation` uses the angular-spectrum method with an `@lru_cache`'d `kzf` grid keyed on `hash(xi.tobytes())` and `hash(yi.tobytes())` — repeated calls with the same grid reuse the cached wavevector array.

5. **Cross-sections** ([crosssections.py](crosssections.py)): `crosssection_xy` (2D focal plane), `crosssection_x` (1D linecut, uses `yi = [0]`), and `crosssection_xz` (x-z stack via a z-loop over `crosssection_x`). All three build the source grid from `params["D"]` and `params["RESOLUTION_X"]`, call `rays2elec2d`, optionally pre-propagate by `params["zfi"]`, pass through `calc_field_after_lens`, then crop to `params["extent_f"]`. `crosssection_xy` also supports a `params["pinhole"]` dict that masks the focal plane and propagates by `zf_pinhole`.

6. **EOM sequences** ([sequences.py](sequences.py)): `phase_freq_func_from_sequence(t_arr, frequency_arr, amp_arr)` builds piecewise phase/frequency/amplitude waveforms from segment lists, preserving phase continuity across segments. `eom_model` wraps this into a `phase_amp_func` suitable for injection into `vipa_rays` via `params["phase_amp_func"]`. The EOM model offsets time per ray by `(nx + ny/FSR_Ratio) * Lrt / c` to account for the ray's round-trip travel through the VIPA. Time series are animated into GIFs and saved as `.npz`.

7. **Detector response** ([sensor.py](sensor.py)): `convolve_detector_response` applies (or with `deconvolve=True` inverts, using Tikhonov regularization) a complex frequency-domain transfer function `H(f)` to a uniformly-sampled time-domain signal.

## Parameter conventions

- All lengths are SI (metres). Wavelength is `params["lambda"]` in metres.
- `params["D"]` = full source-plane extent; `RESOLUTION_X` = source-plane pixel size; source grid is `N_grid = int(D/(2*RESOLUTION_X))*2 + 1` on each axis (always odd).
- `params["extent_f"]` is used only to crop (and label plots of) the focal-plane output — it does not affect the FFT resolution. Focal-plane resolution is set by `D`; focal-plane extent is set by `RESOLUTION_X`.
- `FSR_Ratio` is the ratio of the along-dispersion to cross-dispersion free spectral ranges, controlling inter-beam phase `-(ix*FSR_Ratio + iy)*phi` and EOM time-of-flight delay.
- Two canonical parameter dicts live in [vipa_focus.py](vipa_focus.py): `PARAMS_100` (100x100 VIPA grid, demo) and `PARAMS_10` (1x9, matches the experimental system).

## Working with the code

- When modifying ray-generation or adding modulation effects, plug in via `params["phase_amp_func"]` and/or `params["displacement_func"]` rather than editing `rays2elec2d` — these callables are the supported extension points.
- `freespace_propagation` caches `kzf` by hashing `xi.tobytes()`; if you mutate `xi`/`yi` in place the cache key changes automatically, but avoid constructing many one-off grids in a loop (cache will thrash).
- The `TYPE = N` branching in `__main__` blocks is how past experiments are preserved — prefer adding a new `TYPE` branch rather than deleting old ones unless the user asks.
