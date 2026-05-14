# VIPA Focus Simulation

The repository contains code to numerically simulate RIPA focusing pattern using fourier transform.

## Vectorial high-NA focusing

`src/vector_focus.py` extends the scalar Fourier lens model in `src/core.py`
to polarized high-NA fields. The general entry point is
`focus_pupil_vectorial(...)`: pass any scalar pupil field as a scalar, array, or
callable `scalar_pupil(Xi, Yi)`, and pass the input Jones field as a constant
vector, array/callable map, or one of the named polarizations `"x"`, `"y"`,
`"left_circular"`, `"right_circular"`, `"radial"`, or `"azimuthal"`.

The Gaussian case is just a convenience wrapper:

```python
from src.vector_focus import (
    focus_pupil_vectorial,
    gaussian_pupil,
    polarization_observables,
)

pupil = gaussian_pupil(0.85 * f * NA / n)
(xf, yf), E_focus, E_pupil = focus_pupil_vectorial(
    wavelength=780e-9,
    NA=0.8,
    f=3e-3,
    n=1.0,
    scalar_pupil=pupil,
    polarization="x",
)
obs = polarization_observables(E_focus)
```

`E_focus` and `E_pupil` have shape `(3, len(y), len(x))` in component order
`Ex, Ey, Ez`. The implementation projects the input Jones pupil through the
Richards-Wolf high-NA basis, then reuses `core.calc_field_after_lens`
component-by-component.
