"""Self-test of the 1D/2D Gaussian fitting utilities (was fit_gaussian.py __main__)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.fit_gaussian import (
    gaussian_1d_offset,
    gaussian_2d,
    fit_gaussian_1d,
    fit_and_plot_gaussian_2d,
)


if __name__ == "__main__":
    # test 1D gaussian fitting
    x = np.linspace(-5, 5, 100)
    mu_true = 0.5
    sigma_true = 1.0
    scale_true = 2.0
    offset_true = 0.5
    y = gaussian_1d_offset(x, mu_true, sigma_true, scale_true, offset_true)
    popt = fit_gaussian_1d(x, y, offset=True)
    print("Fitted parameters: ", popt)
    plt.plot(x, y, label="Data")
    y_fit = gaussian_1d_offset(x, *popt)
    plt.plot(x, y_fit, "--", label="Gaussian fit")
    plt.legend()
    plt.title("1D Gaussian Fit Test")
    plt.xlabel("x")
    plt.ylabel("Intensity")
    plt.show()
    # test 2D gaussian fitting
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    cov = [[1, 0], [0, 1]]
    sigma = np.sqrt(np.linalg.eigvals(cov))
    print("Sigma: ", sigma)
    Z = gaussian_2d(X, Y, mu=[0, 0], cov=cov)
    popt = fit_and_plot_gaussian_2d(X, Y, Z)
    print(popt)
    plt.show()
