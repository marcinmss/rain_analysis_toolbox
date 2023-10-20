from stereo3d import Stereo3DSeries
from typing import Tuple, List, Any
from stereo3d.stereo3d_dataclass import MINDIST, MAXDIST, BASEAREASTEREO3D
from numpy import arange, zeros, divide, array, ndarray
from aux_funcs.calculations_for_parsivel_data import volume_drop

EPISILON = 10e-9


def area_of_session(session_limits: Tuple[float, float]) -> float:
    left, right = session_limits
    assert left < right, " The left bound has to be bigger than the right!"
    assert 200.0 <= left, "The left bound can't be smaller than 200"
    assert right <= 400.0, "The right bound can't be bigger than 400"

    return BASEAREASTEREO3D * (right**2 - left**2) / (MAXDIST**2 - MINDIST**2)


"""
w
"""


def filter_by_distance_to_sensor(
    series: Stereo3DSeries, new_limits: Tuple[float, float]
) -> Stereo3DSeries:
    new_area = area_of_session(new_limits)
    left, right = new_limits

    return Stereo3DSeries(
        series.duration,
        array([item for item in series if left <= item.distance_to_sensor <= right]),
        new_area,
    )


def acumulate_by_distance(
    series: Stereo3DSeries, N: int = 1024
) -> List[ndarray[float, Any]]:
    length = (MAXDIST - MINDIST) / N
    volume = zeros(shape=(N,), dtype=float)
    mean_diameter = zeros(shape=(N,), dtype=float)
    mean_velocity = zeros(shape=(N,), dtype=float)
    number_drops = zeros(shape=(N,), dtype=float)

    # Calculate the area of each session
    areas = [area_of_session((b, b + length)) for b in arange(200.0, 400.0, length)]

    for row in series:
        idx = int((row.distance_to_sensor - MINDIST - EPISILON) // length)
        mean_diameter[idx] += row.diameter
        mean_velocity[idx] += row.velocity
        number_drops[idx] += 1

        volume[idx] += volume_drop(row.diameter)

    ndrops_by_area = divide(number_drops, areas, where=(areas != 0))
    mean_diameter = divide(mean_diameter, number_drops, where=(number_drops != 0))
    mean_velocity = divide(mean_velocity, number_drops, where=(number_drops != 0))

    return [volume, ndrops_by_area, number_drops, mean_diameter, mean_velocity]
