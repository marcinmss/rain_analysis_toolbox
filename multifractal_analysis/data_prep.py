from numpy import nan_to_num, nanmean, ndarray, zeros

from multifractal_analysis.split_array import (
    slice_to_clossest_smaller_power_of_2,
    slice_to_power_of_2,
)

"""
Function for reshaping and normalizing one dimensional data
"""


def prep_data(
    field_1d: ndarray, size: int | None = None, fluc: bool = False
) -> ndarray:
    output = nan_to_num(field_1d.flatten())
    if size is None:
        output = slice_to_clossest_smaller_power_of_2(output)
    else:
        output = slice_to_power_of_2(field_1d, size)

    if fluc:
        output = fluctuations(output)

    return output.reshape((-1, 1)) / nanmean(output)


def prep_data_ensemble(field_1d: ndarray, size: int, fluc: bool = False) -> ndarray:
    output = slice_to_clossest_smaller_power_of_2(field_1d).reshape((size, -1))
    output = nan_to_num(output)
    if fluc:
        output = fluctuations(output)
    output = output / nanmean(output)
    return output


"""
Function for getting the fluctuations of an field
"""


def fluctuations(field: ndarray) -> ndarray:
    match field.ndim:
        case 1:
            return _fluctuations_1d(field)
        case 2:
            output = zeros(field.shape, dtype=float)
            for i in range(output.shape[1]):
                output[:, i] = _fluctuations_1d(field[:, i])
            return output
        case _:
            assert False, "The array must be 1 or 2 dimensional"


def _fluctuations_1d(field_1d: ndarray) -> ndarray:
    field_fluct = zeros(field_1d.shape, dtype=float)
    field_fluct[:-1] = [
        abs(field_1d[i + 1] - field_1d[i]) for i in range(field_1d.size - 1)
    ]
    field_fluct[-1] = field_fluct[-2]

    return field_fluct
