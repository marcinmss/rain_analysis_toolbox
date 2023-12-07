from numpy import ndarray, log
from typing import Any, Generator, Tuple


"""
Function to find the closet smaller power of two
"""


def closest_smaller_power_of_2(number: int) -> int:
    n = 1
    while 2 * n < number:
        n = 2 * n
    return n


"""
Function to check if an array is a power of 2
"""


def is_power_of_2(number: int) -> bool:
    return closest_smaller_power_of_2(number) == number


"""
Function to generate from a single array, others arrays at diferent scalles
"""


def upscale(field: ndarray) -> Generator[Tuple[int, ndarray], Any, Any]:
    lamb = field.shape[0]
    assert is_power_of_2(lamb), "Array needs to be a power of 2"

    averaged_field = field.copy()
    while lamb > 0:
        yield (lamb, averaged_field)
        averaged_field = (averaged_field[::2, :] + averaged_field[1::2, :]) / 2
        lamb //= 2


"""
Function for the theoretical value for k acording to the UM model
"""


def kq_theoretical(q: float, alpha: float, c1: float, h: float) -> float:
    # The output is the values of the scaling moment function
    if alpha == 1:
        return c1 * q * log(q) + h * q
    else:
        return c1 * (q**alpha - q) / (alpha - 1) + h * q
