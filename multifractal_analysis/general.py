from numpy import ndarray, power, sum as npsum, mean, log, zeros
from typing import Any, Generator, Tuple, List

"""
Function for spliting a field into multiple smaller field power of 2
"""


def split_field(field_1d: ndarray, power_of_2: int, threshold: float) -> List[ndarray]:
    field_1d = pad_to_power_of_2(field_1d)
    stack = [field_1d]
    output = []
    final_lenght = 2**power_of_2
    while len(stack) > 0:
        curr_field = stack.pop()
        if curr_field.size == final_lenght:
            output.append(curr_field)
        else:
            stack.extend(analyse(curr_field, threshold))
    return output


def analyse(field_1d: ndarray, threshold: float) -> List[ndarray]:
    n_2 = field_1d.size // 2
    first_half = field_1d[:n_2]
    second_half = field_1d[n_2:]
    if npsum(first_half) > threshold and npsum(second_half) > threshold:
        return [first_half, second_half]
    else:
        best_half = max((field_1d[i : i + n_2] for i in range(n_2)), key=npsum)
        if npsum(best_half) > threshold:
            return [best_half]
        else:
            return []


"""
Function for getting the fluctuations of an field
"""


def fluctuations(field_1d: ndarray) -> ndarray:
    field_fluct = zeros(field_1d.shape, dtype=float)
    field_fluct[:-1] = [
        abs(field_1d[i + 1] - field_1d[i]) for i in range(field_1d.size - 1)
    ]
    field_fluct[-1] = field_fluct[-2]

    return field_fluct


"""
Function for the theoretical value for k acording to the UM model
"""


def kq_theoretical(q: float, alpha: float, c1: float, h: float) -> float:
    # The output is the values of the scaling moment function
    if alpha == 1:
        return c1 * q * log(q) + h * q
    else:
        return c1 * (q**alpha - q) / (alpha - 1) + h * q


"""
Function to check if an array is a power of 2
"""


def slice_power_of_2(field: ndarray) -> ndarray:
    n = closest_power_of_2(field.size)
    possible_arrays = [field[i : i + n] for i in range(field.size - n)]
    return max(possible_arrays, key=npsum)


def pad_to_power_of_2(field: ndarray) -> ndarray:
    n = field.size
    n2 = closest_power_of_2(field.size * 2)
    if n == n2:
        return field
    output = zeros(n2, dtype=float)
    output[n2 // 2 - n // 2 : n2 // 2 + (n - n // 2)] = field[:]
    return output


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
Function to find the closset smaller power of two
"""


def closest_power_of_2(length_array: int) -> int:
    n = 1
    while 2 * n < length_array:
        n = 2 * n
    return n


"""
Function to generate from a single array, others arrays at diferent scalles
"""


def upscale(field_1d: ndarray) -> Generator[Tuple[int, ndarray], Any, Any]:
    lamb = field_1d.shape[0]
    assert is_power_of_2(lamb), "Array needs to be a power of 2"

    averaged_field = field_1d
    while lamb > 0:
        yield (lamb, averaged_field)
        averaged_field = (averaged_field[::2] + averaged_field[1::2]) / 2
        lamb //= 2


"""
Function for calculating the moment of a field
"""


def moment(field_1d: ndarray, q: float) -> float:
    output = mean(power(field_1d, q))
    assert isinstance(output, float)
    return output
