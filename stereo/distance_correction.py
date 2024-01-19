from typing import List
from aux_funcs.calculations_for_parsivel_data import volume_drop
from stereo.distance_analisys import area_of_session, volume_of_session
from stereo.dataclass import MINDIST, MAXDIST, Stereo3DRow, Stereo3DSeries
from numpy import (
    argmax,
    array,
    exp,
    linspace,
    log,
    ndarray,
    zeros,
)
from itertools import pairwise

NCLASSESDIAM = 50
MAXDIAM = 25
MINDIAM = 0.0
RESOL_DIAMETER = (MAXDIAM - MINDIAM) / NCLASSESDIAM
BINSDIAMETER = array(
    [pair for pair in pairwise(linspace(MINDIAM, MAXDIAM, NCLASSESDIAM + 1))]
)

"""
Create a function for getting the class of diameter
"""


def get_diameter_class(diameter: float) -> int:
    return int((diameter - MINDIAM - 1e-9) // RESOL_DIAMETER)


NCLASSESDIST = 10
RESOL_DIST = (MAXDIST - MINDIST) / NCLASSESDIST
BINSDIST = array(
    [pair for pair in pairwise(linspace(MINDIST, MAXDIST, NCLASSESDIST + 1))]
)
AREAS = array([area_of_session(limit) for limit in BINSDIST])
VOLUMES = array([volume_of_session(limit) for limit in BINSDIST])


def get_distance_class(distance: float) -> int:
    return int((distance - MINDIST - 1e-9) // RESOL_DIST)


def amortizacao(value: float) -> float:
    a = -0.3
    m = 2
    c = log(m - 1) - a
    return m - exp(a * value + c)


def get_correction_map(series: Stereo3DSeries | List[Stereo3DRow]) -> ndarray:
    density_map = zeros((NCLASSESDIST, NCLASSESDIAM), dtype=float)
    for row in series:
        distclass = get_distance_class(row.distance_to_sensor)
        diamclass = get_diameter_class(row.diameter)
        density_map[distclass, diamclass] += 1 / VOLUMES[distclass]

    argmax_density = argmax(density_map, axis=0)

    correction_map = zeros((NCLASSESDIST, NCLASSESDIAM), dtype=float)
    for diamclass in range(NCLASSESDIAM):
        maximum_density = argmax_density[diamclass]
        volume_max_density = VOLUMES[argmax_density[diamclass]]
        if maximum_density == 0:
            continue
        for distclass in range(NCLASSESDIST):
            if density_map[distclass, diamclass] == 0:
                continue
            value = (maximum_density * volume_max_density) / (
                density_map[distclass, diamclass] * VOLUMES[distclass]
            )

            correction_map[distclass, diamclass] = amortizacao(value)

    return correction_map


def compute_corrected_rain_rate(
    series: Stereo3DSeries, interval_seconds: int
) -> ndarray:
    # Define the ends of the time series
    start, stop = series.duration

    # Correction map
    correction_map = get_correction_map(series)

    # Create an empty object with the slots to fit the data
    rain_rate = zeros(shape=((stop - start) // interval_seconds + 1,), dtype=float)

    # Loop thought every row of data and add the rate until you have a value
    factors = []
    for row in series:
        idx = (row.timestamp - start) // interval_seconds
        distclass = get_distance_class(row.distance_to_sensor)
        diamclass = get_diameter_class(row.diameter)
        factor = correction_map[distclass, diamclass]
        factors.append(factor)
        rain_rate[idx] += (
            volume_drop(row.diameter)
            * factor
            / series.area_of_study
            / (interval_seconds / 3600)
        )

    if all([factor == 0 for factor in factors]):
        print("All factors are zeros.")
    print(correction_map)
    return rain_rate
