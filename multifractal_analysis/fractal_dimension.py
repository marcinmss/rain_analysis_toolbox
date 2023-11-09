from typing import Any, Generator, Tuple
from numpy import empty, log2, ndarray
from multifractal_analysis.regression_solution import RegressionSolution

"""
Function to check if an array is a power of 2
"""


def is_power_of_2(length_array: int) -> bool:
    if length_array == 1:
        return True
    elif length_array % 2 != 0:
        return False
    else:
        return is_power_of_2(length_array // 2)


"""
Function to generate from a single array, others arrays at diferent scalles
"""


def walk_scalles(field_1d: ndarray) -> Generator[Tuple[int, ndarray], Any, Any]:
    lamb = field_1d.shape[0]
    assert is_power_of_2(lamb), "Array needs to be a power of 2"

    averaged_field = field_1d
    while lamb > 1:
        yield (lamb, averaged_field)
        averaged_field = (averaged_field[::2] + averaged_field[1::2]) / 2
        lamb //= 2


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

    for i, (lamb, scalled_array) in enumerate(walk_scalles(field_1d)):
        threshold = lamb**gamma
        boxcount = count_boxes(scalled_array > threshold)
        x[i], y[i] = log2(lamb), log2(boxcount)

    return (x, y)


def get_fractal_dimension(field_1d: ndarray, gamma: float) -> RegressionSolution:
    x, y = get_fractal_dimension_points(field_1d, gamma)
    return RegressionSolution(x, y)
