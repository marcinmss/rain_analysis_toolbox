from typing import Tuple
from numpy import log, ndarray, absolute, arange, zeros
from collections import namedtuple
from matplotlib.axes import Axes
from numpy.fft import fft
from multifractal_analysis.general import is_power_of_2, kq_theoretical
from multifractal_analysis.regression_solution import RegressionSolution
from multifractal_analysis.double_trace_moment import get_um_params_dtm


"""
Produces the points for the spectral analysis
"""


def get_spectral_analysis_points(field: ndarray) -> Tuple[ndarray, ndarray]:
    assert is_power_of_2(field.shape[0]), "The field needs to be a power of 2"
    n = field.shape[0] // 2
    n_samples = field.shape[1]
    E = zeros(n, dtype=float)
    for sample in range(n_samples):
        buf = absolute(fft(field[:, sample]))[0:n]
        E += (buf * buf / n_samples).flatten()
    k = arange(1.0001, n + 1.0001, 1)
    return (log(k), log(E))


"""
Calculate the beta and H coeficients
"""


def get_beta_and_h(field_1d: ndarray) -> Tuple[float, float]:
    assert is_power_of_2(field_1d.shape[0]), "The field needs to be a power of 2"
    x, y = get_spectral_analysis_points(field_1d)
    regression_line = RegressionSolution(x, y)
    beta = -regression_line.angular_coef
    alpha, c1 = get_um_params_dtm(field_1d, 1.5)
    k_2 = kq_theoretical(2, alpha, c1, 0)
    h = 0.5 * (beta - 1 + k_2)
    return (beta, h)


"""
Do the whole spectral analysis an plot the results
"""
SpectralAnalysis = namedtuple("SpectralAnalysis", ["beta", "h", "rsquare", "points"])


def spectral_analysis(field: ndarray, ax: Axes | None = None) -> SpectralAnalysis:
    assert is_power_of_2(field.shape[0]), "The field needs to be a power of 2"

    # First get the spectral analysis points
    x, y = get_spectral_analysis_points(field)
    regression_line = RegressionSolution(x, y)
    rsquare = regression_line.r_square
    beta = -regression_line.angular_coef
    alpha, c1 = get_um_params_dtm(field, 1.5)
    k_2 = kq_theoretical(2, alpha, c1, 0)
    h = 0.5 * (beta - 1 + k_2)

    # Now if there is an axis, plot the results in it
    leftx, rightx = min(x), max(x)
    if ax is not None:
        ax.set_title("Spectral Analysis")
        ax.set_ylabel(r"$\log _2 (E)$")
        ax.set_xlabel(r"$\log _2 (k)$")
        ax.scatter(x, y, marker="x", c="k")
        # Plot the best fit line
        a, b = regression_line.angular_coef, regression_line.linear_coef
        ax.plot((leftx, rightx), (a * leftx + b, a * rightx + b), c="red")

        # Plot the text with the results
        textstr = ", ".join(
            (r"$\beta=$%.2f" % (beta,), r"$h=$%.2f" % (h,), r"$r^2=%.2f$" % (rsquare,))
        )

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.0)

        ax.text(
            0.55,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

    return SpectralAnalysis(beta, h, rsquare, (x, y))
