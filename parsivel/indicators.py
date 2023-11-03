from aux_funcs.calculations_for_parsivel_data import volume_drop
from parsivel.parsivel_dataclass import ParsivelTimeSeries
from numpy import ndarray, sum as npsum
from parsivel.matrix_classes import (
    CLASSES_DIAMETER_MIDDLE,
    CLASSES_VELOCITY_MIDDLE,
    CLASSES_DIAMETER_BINS,
)

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
    return (CLASSES_DIAMETER_BINS, ndrops)
