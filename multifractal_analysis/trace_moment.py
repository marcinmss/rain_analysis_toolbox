from math import log, log2
from matplotlib import colormaps
from typing import Tuple
from numpy import fromiter, linspace, ndarray, mean, power, zeros
from matplotlib.axes import Axes
from multifractal_analysis.general import is_power_of_2, upscale
from multifractal_analysis.regression_solution import RegressionSolution
from collections import namedtuple

CMAP = colormaps["Set1"]

"""
Function for calculating the moment of a field
"""


def moment_tm(field: ndarray, q: float) -> float:
    return mean(power(field, q), dtype=float)


"""
Function for getting the TM points
"""


def get_trace_moment_points(field: ndarray, q: float = 1.0) -> Tuple[ndarray, ndarray]:
    outer_scale = int(log2(field.shape[0])) + 1
    x, y = zeros(outer_scale, dtype=float), zeros(outer_scale, dtype=float)
    for i, (lamb, scalled_array) in enumerate(upscale(field)):
        x[i], y[i] = log(lamb), log(moment_tm(scalled_array, q))

    return (x, y)


def k_of_q(field_1d: ndarray, q: float = 1.5):
    x, y = get_trace_moment_points(field_1d, q)
    return RegressionSolution(x, y).angular_coef


"""
Calculate C1 and alpha
"""


def get_um_params_tm(field_1d: ndarray):
    h = 0.1
    kf = k_of_q(field_1d, 1.0 + h)
    k0 = k_of_q(field_1d, 1.0 - h)
    c1 = (kf - k0) / (2 * h)
    alpha = (kf + k0) / (h**2 * c1)
    return alpha, c1


"""
Function for running the whole TM analysis including plotting the graph
"""
TMAnalysis = namedtuple("TMAnalysis", ["alpha", "c1", "r_square"])


def tm_analysis(field: ndarray, ax: Axes | None = None) -> TMAnalysis:
    assert is_power_of_2(field.shape[0]), "The field needs to be a power of 2"

    if ax is not None:
        for q in reversed((0.1, 0.5, 0.8, 1.01, 1.5, 2.0, 2.5)):
            # Get the points and do the regression on them
            x, y = get_trace_moment_points(field, q=q)

            # Set the axis apperence
            ax.set_title("TM Analysis")
            ax.set_ylabel(r"$\log (TM_\lambda)$")
            ax.set_xlabel(r"$\log (\lambda)$")
            color = CMAP(q / 2.5)

            # Plot the tendency line for each analysis
            regression_solution = RegressionSolution(x, y)
            a, b = regression_solution.angular_coef, regression_solution.linear_coef
            leftx, rightx = min(x), max(x)
            ax.plot((leftx, rightx), (a * leftx + b, a * rightx + b), c=color)

            # Plot the points of each analysis
            legend = ", ".join(
                (
                    r"$q=%.2f$" % (q,),
                    r"$r^2=%.2f$" % (regression_solution.r_square,),
                )
            )
            ax.scatter(x, y, color=color, label=legend, edgecolors="k")
            ax.legend(prop={"size": 6}, framealpha=0.0, loc="upper left")
            yb1, yb2 = ax.get_ybound()
            ax.set_ybound(yb1, (yb2 - yb1) * 1.4 + yb1)

    x, y = get_trace_moment_points(field, q=1.5)
    r_square = RegressionSolution(x, y).r_square

    alpha, c1 = get_um_params_tm(field)
    return TMAnalysis(alpha, c1, r_square)


"""
Function for plotting the empirical k_of_q
"""


def empirical_k_of_q(field: ndarray, ax: Axes | None = None):
    if ax is not None:
        # Set the axis apperence
        ax.set_title("Empirical $K(q)$")
        ax.set_ylabel(r"$K(q)$")
        ax.set_xlabel(r"$q$")
        x = linspace(0.1, 3.0, 15)
        y = fromiter((k_of_q(field, q) for q in x), dtype=float)
        ax.plot(x, y, "k")
