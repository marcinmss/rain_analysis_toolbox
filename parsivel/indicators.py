from typing import Literal, Tuple
from aux_funcs.general import volume_drop
from parsivel.dataclass import ParsivelTimeSeries
from numpy import ndarray, sum as npsum, zeros
from parsivel.matrix_classes import (
    CLASSES_DIAMETER,
    CLASSES_DIAMETER_MIDDLE,
    CLASSES_VELOCITY_MIDDLE,
)

"""
Given a standard 32x32 Parsivel matrix, calculates the aproximate total volume
using the equivalent diameter
"""


def matrix_to_volume(matrix: ndarray | Literal[0]) -> float:
    n_per_diameter = npsum(matrix, axis=0)
    return sum(
        [n * volume_drop(d) for (n, d) in zip(n_per_diameter, CLASSES_DIAMETER_MIDDLE)]
    )


def mean_diameter_matrix(matrix: ndarray | Literal[0]) -> float:
    ndrops = 0.0
    sum_diameters = 0.0
    for diam, nd in zip(CLASSES_DIAMETER_MIDDLE, npsum(matrix, axis=0)):
        ndrops += nd
        sum_diameters += diam * nd

    return sum_diameters / ndrops if ndrops > 0.0 else 0.0


"""
Calculate the mean diameter and velocity
"""


def get_mean_diameter(series: ParsivelTimeSeries) -> float:
    matrix_for_event = series.matrix_for_event
    ndrops = 0.0
    sum_diameters = 0.0
    for diam, nd in zip(CLASSES_DIAMETER_MIDDLE, npsum(matrix_for_event, axis=0)):
        ndrops += nd
        sum_diameters += diam * nd

    return sum_diameters / ndrops if ndrops > 0.0 else 0.0


def get_mean_velocity(series: ParsivelTimeSeries) -> float:
    matrix_for_event = series.matrix_for_event
    ndrops = 0.0
    sum_velocitys = 0.0
    for diam, nd in zip(CLASSES_VELOCITY_MIDDLE, npsum(matrix_for_event, axis=1)):
        ndrops += nd
        sum_velocitys += diam * nd

    return sum_velocitys / ndrops if ndrops > 0.0 else 0.0


"""
Calculate the mean diameter and velocity
"""


def get_kinetic_energy(series: ParsivelTimeSeries) -> float:
    total_kinetic_energy = 0.0
    event_matrix = series.matrix_for_event
    assert isinstance(event_matrix, ndarray)
    for i in range(32):
        mass_drop = volume_drop(CLASSES_DIAMETER_MIDDLE[i]) * 1e-9
        for j in range(32):
            velocity_drop = CLASSES_VELOCITY_MIDDLE[j]
            total_kinetic_energy += (
                mass_drop * (velocity_drop**2) / 2 * event_matrix[i, j]
            )

    return total_kinetic_energy


"""
Get the drop size distribution for a parsivel series
"""


def get_ndrops_in_each_diameter(series: ParsivelTimeSeries):
    ndrops = npsum(series.matrix_for_event, axis=0)
    return (CLASSES_DIAMETER_MIDDLE, ndrops / series.area_of_study)


"""
Get the (d, n(d)) for a series
"""


def get_nd(series: ParsivelTimeSeries) -> Tuple[ndarray, ndarray]:
    # Get relevant variables
    matrix = series.matrix_for_event
    assert isinstance(matrix, ndarray)
    dt = series.duration[1] - series.duration[0]
    area_m2 = series.area_of_study * 1e-6

    # Calculate the data
    y = zeros(32, dtype=float)
    x = zeros(32, dtype=float)
    for idx_diam in range(32):
        mult_factor = 1 / (area_m2 * dt * CLASSES_DIAMETER[idx_diam][1])
        x[idx_diam] = CLASSES_DIAMETER_MIDDLE[idx_diam]
        y[idx_diam] = (
            sum(
                matrix[idx_vel, idx_diam] / CLASSES_VELOCITY_MIDDLE[idx_vel]
                for idx_vel in range(32)
            )
            * mult_factor
        )

    return (x, y)


"""
Get the (d, n(d)*d^3) for a series
"""


def get_nd3(series: ParsivelTimeSeries) -> Tuple[ndarray, ndarray]:
    d, nd = get_nd(series)
    return (d, nd * d**3)
