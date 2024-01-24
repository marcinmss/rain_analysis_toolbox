from itertools import pairwise
from stereo import Stereo3DSeries
from typing import Tuple, List, Any
from stereo.dataclass import MINDIST, MAXDIST, BASEAREASTEREO3D
from numpy import arange, cos, linspace, pi, zeros, divide, array, ndarray
from aux_funcs.general import volume_drop

EPISILON = 10e-9
CONEANGLE = BASEAREASTEREO3D / (MAXDIST**2 - MINDIST**2) * 2
SOLIDANGLE = 2 * pi * (1 - cos(CONEANGLE / 2))

"""
Function for calculating the area of study of the series. 
Used for calculating matrics such as rain rate
"""


def area_of_session(session_limits: Tuple[float, float]) -> float:
    left, right = session_limits
    assert left < right, " The left bound has to be bigger than the right!"
    assert 200.0 <= left, "The left bound can't be smaller than 200"
    assert right <= 400.0, "The right bound can't be bigger than 400"

    return CONEANGLE * (right**2 - left**2) / 2


def volume_of_session(session_limits: Tuple[float, float]) -> float:
    left, right = session_limits
    assert left < right, " The left bound has to be bigger than the right!"
    assert 200.0 <= left, "The left bound can't be smaller than 200"
    assert right <= 400.0, "The right bound can't be bigger than 400"

    return (right**3 - left**3) / 3 * SOLIDANGLE


"""
Function for getting only the drops withing a certain range of the sensor
"""


def filter_by_distance_to_sensor(
    series: Stereo3DSeries, new_limits: Tuple[float, float]
) -> Stereo3DSeries:
    left, right = new_limits

    return Stereo3DSeries(
        f"stereo3d({left:.1f}:{right:.1f})",
        series.duration,
        array([item for item in series if left <= item.distance_to_sensor <= right]),
        (new_limits),
    )


"""
Split the Stereo3DSeries into multiple series based on the distance to the sensor
"""


def split_by_distance_to_sensor(
    series: Stereo3DSeries, number_of_sessions: int = 8
) -> List[Stereo3DSeries]:
    left_lim, right_lim = series.limits_area_of_study
    assert number_of_sessions > 0, "The number of sessions needs to be bigger than zero"
    output = [
        filter_by_distance_to_sensor(series, new_lim)
        for new_lim in pairwise(linspace(left_lim, right_lim, number_of_sessions + 1))
    ]
    return output


"""
Getting certains metrics based on the distance to the sensor
"""


def acumulate_by_distance(
    series: Stereo3DSeries, N: int = 1024
) -> List[ndarray[float, Any]]:
    length = (MAXDIST - MINDIST) / N
    total_volume = zeros(shape=(N,), dtype=float)
    mean_diameter = zeros(shape=(N,), dtype=float)
    mean_velocity = zeros(shape=(N,), dtype=float)
    number_drops = zeros(shape=(N,), dtype=float)

    # Calculate the area of each session
    areas = [area_of_session((b, b + length)) for b in arange(200.0, 400.0, length)]
    volumes = [volume_of_session((b, b + length)) for b in arange(200.0, 400.0, length)]

    for row in series:
        idx = int((row.distance_to_sensor - MINDIST - EPISILON) // length)
        mean_diameter[idx] += row.diameter
        mean_velocity[idx] += row.velocity
        number_drops[idx] += 1

        total_volume[idx] += volume_drop(row.diameter)

    total_volume_by_area = divide(total_volume, areas, where=(areas != 0))
    total_volume_by_volume = divide(total_volume, volumes, where=(volumes != 0))
    ndrops_by_area = divide(number_drops, areas, where=(areas != 0))
    ndrops_by_volume = divide(number_drops, volumes, where=(volumes != 0))
    mean_diameter = divide(mean_diameter, number_drops, where=(number_drops != 0))
    mean_velocity = divide(mean_velocity, number_drops, where=(number_drops != 0))

    return [
        number_drops,
        ndrops_by_area,
        ndrops_by_volume,
        total_volume,
        total_volume_by_area,
        total_volume_by_volume,
        mean_diameter,
        mean_velocity,
    ]
