from typing import Tuple
from numpy import empty, log2, ndarray
from matplotlib.axes import Axes
from multifractal_analysis.regression_solution import RegressionSolution
from multifractal_analysis.general import is_power_of_2, upscale
from collections import namedtuple


"""
Function to do the box counting
"""


def count_boxes(data: ndarray) -> int:
    return sum(1 for item in data.flatten() if item)


"""
Get the fractal dimension
"""


def get_fractal_dimension_points(field_1d: ndarray) -> Tuple[ndarray, ndarray]:
    outer_scale = int(log2(field_1d.shape[0])) + 1
    x, y = empty(outer_scale, dtype=float), empty(outer_scale, dtype=float)

    for i, (lamb, scalled_array) in enumerate(upscale(field_1d)):
        threshold = 0
        boxcount = count_boxes(scalled_array > threshold)
        x[i], y[i] = log2(lamb), log2(boxcount) if boxcount > 0 else 0

    return (x, y)


def get_fractal_dimension(field_1d: ndarray) -> RegressionSolution:
    x, y = get_fractal_dimension_points(field_1d)
    return RegressionSolution(x, y)


"""
Function for doing the fractal dimension analysis
"""
FractalDimensionAnalysis = namedtuple("FractalDimensionAnalysis", ["df"])


def fractal_dimension_analysis(
    field: ndarray, ax: Axes | None = None
) -> FractalDimensionAnalysis:
    assert is_power_of_2(field.shape[0]), "The field needs to be a power of 2"

    # Get the points for the analysis
    x, y = get_fractal_dimension_points(field)
    regression_line = RegressionSolution(x, y)
    df = regression_line.angular_coef

    # Plot the graph if there is an axes
    if ax is not None:
        # Set the axis apperence
        ax.set_title("Fractal Dimension Analysis")
        ax.set_ylabel(r"$\log _2 (N_\lambda)$")
        ax.set_xlabel(r"$\log _2 (\lambda)$")

        # Plot the line showing the best line chosen to calculate alpha
        text = ", ".join(
            (r"$D_f=$%.2f" % (df,), r"$r^2=$%.2f" % (regression_line.r_square,))
        )
        a, b = regression_line.angular_coef, regression_line.linear_coef
        xmin, xmax = 0.0, max(regression_line.xpoints)
        ax.plot((xmin, xmax), (a * xmin + b, a * xmax + b), c="k", label=text)
        ax.legend(prop={}, framealpha=0.0)

        # Plot the points of each analysis
        ax.scatter(x, y, marker="x")

    return FractalDimensionAnalysis(df)
