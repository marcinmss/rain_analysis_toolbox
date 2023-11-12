from numpy import ndarray
from typing import Any, Generator, Tuple

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


def upscale(field_1d: ndarray) -> Generator[Tuple[int, ndarray], Any, Any]:
    lamb = field_1d.shape[0]
    assert is_power_of_2(lamb), "Array needs to be a power of 2"

    averaged_field = field_1d
    while lamb > 1:
        yield (lamb, averaged_field)
        averaged_field = (averaged_field[::2] + averaged_field[1::2]) / 2
        lamb //= 2


"""
Function for calculating the moment of a field
"""


def moment(field_1d: ndarray, q: float) -> float:
    lamb = field_1d.flatten().shape[0]
    if lamb > 0:
        return sum(field_1d**q) / lamb
    else:
        return 0
