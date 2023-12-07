from numpy import ndarray, sum as npsum, zeros
from multifractal_analysis.general import closest_smaller_power_of_2, is_power_of_2


"""
Function to check if an array is a power of 2
"""


def slice_to_clossest_smaller_power_of_2(field_1d: ndarray) -> ndarray:
    n = closest_smaller_power_of_2(field_1d.size)
    possible_arrays = [field_1d[i : i + n] for i in range(field_1d.size - n)]
    return max(possible_arrays, key=npsum)


def slice_to_power_of_2(field: ndarray, size: int) -> ndarray:
    assert (
        size > field.size
    ), "Size need to be smaller than the one of the original array"

    possible_arrays = [field[i : i + size] for i in range(field.size - size)]
    return max(possible_arrays, key=npsum)


def pad_to_power_of_2(field_1d: ndarray) -> ndarray:
    n = field_1d.size
    if is_power_of_2(n):
        return field_1d
    n2 = closest_smaller_power_of_2(field_1d.size * 2)
    output = zeros((n2,), dtype=float)
    output[n2 // 2 - n // 2 : n2 // 2 + (n - n // 2)] = field_1d[:]
    return output
