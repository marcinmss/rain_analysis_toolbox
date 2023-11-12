from typing import Tuple
from numpy import empty, log2, ndarray
from multifractal_analysis.regression_solution import RegressionSolution
from multifractal_analysis.general import upscale


"""
Function to do the box counting
"""


def count_boxes(data: ndarray) -> int:
    return sum(1 for item in data if item)


"""
Get the fractal dimension
"""


def get_fractal_dimension_points(
    field_1d: ndarray, gamma: float
) -> Tuple[ndarray, ndarray]:
    outer_scale = int(log2(field_1d.shape[0]))
    x, y = empty(outer_scale, dtype=float), empty(outer_scale, dtype=float)

    for i, (lamb, scalled_array) in enumerate(upscale(field_1d)):
        threshold = lamb**gamma
        boxcount = count_boxes(scalled_array > threshold)
        x[i], y[i] = log2(lamb), log2(boxcount)

    return (x, y)


def get_fractal_dimension(field_1d: ndarray, gamma: float) -> RegressionSolution:
    x, y = get_fractal_dimension_points(field_1d, gamma)
    return RegressionSolution(x, y)
