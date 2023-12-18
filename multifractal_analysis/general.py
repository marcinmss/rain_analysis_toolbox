from numpy import ndarray, log, log2, floor
from typing import Any, Generator, Tuple

"""
Is power of 2
"""
POWERS_RECORDED = set([1])
MAXIMUM_NUMBER_CHECKED = 1


def is_power_of_2(number: int) -> bool:
    global MAXIMUM_NUMBER_CHECKED, POWERS_RECORDED

    # If is in POWERS_RECORDED return true
    if number in POWERS_RECORDED:
        return True
    elif number < MAXIMUM_NUMBER_CHECKED:
        return False
    else:
        MAXIMUM_NUMBER_CHECKED *= 2
        POWERS_RECORDED.add(MAXIMUM_NUMBER_CHECKED)
        return is_power_of_2(number)


"""
Function to find the closet smaller power of two
"""


def closest_smaller_power_of_2(number: int) -> int:
    return 2 ** int(floor(log2(number)))


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
