from typing import Tuple
from numpy import array, linspace, log2, mean, ndarray, log, log10, empty, argmax, power
from multifractal_analysis.general import is_power_of_2, upscale
from multifractal_analysis.regression_solution import RegressionSolution
from matplotlib.axes import Axes
from collections import namedtuple
from matplotlib import colormaps

CMAP = colormaps["Set1"]

"""
Function for calculating the moment of a field
"""


def moment_dtm1(field_1d: ndarray, q: float) -> float:
    output = mean(power(field_1d, q), dtype=float)
    return output


"""
Function for the moment used in DTM
"""


def moment_dtm2(field: ndarray, q: float) -> float:
    return moment_dtm1(field / mean(field), q)


"""
Functions for finding K(q, eta)
"""


def get_kqeta_points(field: ndarray, q: float, eta: float) -> Tuple[ndarray, ndarray]:
    outer_scale = int(log2(field.shape[0])) + 1
    x, y = empty(outer_scale, dtype=float), empty(outer_scale, dtype=float)
    for i, (lamb, scalled_array) in enumerate(upscale(power(field, eta))):
        x[i], y[i] = log(lamb), log(moment_dtm1(scalled_array, q))

    return (x, y)


def get_kqeta(field: ndarray, q: float, eta: float) -> RegressionSolution:
    x, y = get_kqeta_points(field, q, eta)
    return RegressionSolution(x, y)


"""
Function for getting the K(q, eta) points agains eta for DTM analysis
"""


def get_eta_vs_kqeta_points(field: ndarray, q: float) -> Tuple[ndarray, ndarray]:
    N = 34
    LIMS = (-2.0, 1.0)
    x, y = empty(N, dtype=float), empty(N, dtype=float)
    for i, eta in enumerate(10**i for i in linspace(LIMS[0], LIMS[1], N)):
        kqeta = get_kqeta(field, q, eta).angular_coef
        x[i], y[i] = log10(eta), log10(kqeta)
    return (x, y)


"""
Function for calculating C1 or alpha using DTM analysis
"""


def get_um_params_dtm(field: ndarray, q: float) -> Tuple[float, float]:
    x, y = get_eta_vs_kqeta_points(field, q)
    search_area = array([i for i in range(15, x.size - 4)])
    assert len(search_area) > 5

    lines = [
        RegressionSolution(x[i - 2 : i + 3], y[i - 2 : i + 3]) for i in search_area
    ]
    best_line = lines[argmax([line.angular_coef for line in lines])]

    alpha = best_line.angular_coef
    b = best_line.linear_coef
    c1 = 10**b * (alpha - 1) / (q**alpha - q)
    return (alpha, c1)


def get_um_params_from_points(
    dtm_pts_x: ndarray, dtm_pts_y: ndarray, q: float
) -> Tuple[float, float, RegressionSolution]:
    search_area = array([i for i in range(15, dtm_pts_x.size - 4)])
    assert len(search_area) > 5

    lines = [
        RegressionSolution(dtm_pts_x[i - 2 : i + 3], dtm_pts_y[i - 2 : i + 3])
        for i in search_area
    ]
    best_line = lines[argmax([line.angular_coef for line in lines])]

    alpha = best_line.angular_coef
    b = best_line.linear_coef
    c1 = 10**b * (alpha - 1) / (q**alpha - q)
    return (alpha, c1, best_line)


"""

"""
DTMAnalysis = namedtuple("DTMAnalysis", ["alpha", "c1", "points"])


def dtm_analysis(
    field: ndarray, ax: Axes | None = None, ax2: Axes | None = None
) -> DTMAnalysis:
    assert is_power_of_2(field.shape[0]), "The field needs to be a power of 2"

    # Get the points for doing the analysis
    dtm_x, dtm_y = get_eta_vs_kqeta_points(field, q=1.5)
    alpha, c1, best_line = get_um_params_from_points(dtm_x, dtm_y, 1.5)

    if ax is not None:
        # Set the axis apperence
        ax.set_title("DTM Analysis")
        ax.set_ylabel(r"$\log _{10} (K(q, \eta))$")
        ax.set_xlabel(r"$\log _{10} (\eta)$")

        # Plot the points of each analysis
        text = ", ".join(
            (r"$q=$%.2f" % (1.5,), r"$\alpha=$%.2f" % (alpha,), r"$c1=$%.2f" % (c1,))
        )
        ax.scatter(dtm_x, dtm_y, c="black", s=8.8, label=text)
        ax.legend(prop={}, framealpha=0.0)

        # Plot the line showing the best line chosen to calculate alpha
        a, b = best_line.angular_coef, best_line.linear_coef
        xmin, xmax = best_line.xpoints[0], best_line.xpoints[-1]
        ax.plot((xmin, xmax), (a * xmin + b, a * xmax + b), c="k")

    if ax2 is not None:
        # Set the axis apperence
        ax2.set_title("DTM Analysis ($q = 1.5$)")
        ax2.set_ylabel(r"$\log (DTM_\lambda)$")
        ax2.set_xlabel(r"$\log (\lambda)$")

        for eta in (0.81, 1.23, 1.81, 2.84):
            # Get the point for each eta and get the tendency line
            x, y = get_kqeta_points(field, 1.5, eta)
            regression_line = RegressionSolution(x, y)

            # Plot the tendecy line
            a, b = regression_line.angular_coef, regression_line.linear_coef
            xmin, xmax = regression_line.xpoints[0], regression_line.xpoints[-1]
            ax2.plot((xmin, xmax), (a * xmin + b, a * xmax + b), c="black")

            # Plot the points of each analysis
            text = ", ".join(
                (r"$\eta=$%.2f" % (eta,), r"$r^2=$%.2f" % (regression_line.r_square,))
            )
            color = CMAP(eta / 2.8)
            ax2.scatter(x, y, color=color, label=text, edgecolors="black")
            ax2.legend(prop={"size": 6}, framealpha=0.0)
            yb1, yb2 = ax2.get_ybound()
            ax2.set_ybound(yb1, yb2 * 1.25)

    return DTMAnalysis(alpha, c1, (dtm_x, dtm_y))
