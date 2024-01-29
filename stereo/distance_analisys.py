from itertools import pairwise
from stereo import Stereo3DSeries
from typing import Tuple, List
from stereo.dataclass import MINDIST, MAXDIST, BASEAREASTEREO3D
from numpy import cos, linspace, pi, array

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
    series: Stereo3DSeries, number_of_sections: int = 8
) -> List[Stereo3DSeries]:
    left_lim, right_lim = series.limits_area_of_study
    assert number_of_sections > 1, "The number of sessions needs to be bigger than 1."
    output = [
        filter_by_distance_to_sensor(series, new_lim)
        for new_lim in pairwise(linspace(left_lim, right_lim, number_of_sections + 1))
    ]
    return output
