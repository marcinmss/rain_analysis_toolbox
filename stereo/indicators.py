from typing import Tuple, Any
from numpy import array, linspace, ndarray, zeros, ceil, floor
from parsivel.matrix_classes import CLASSES_DIAMETER_MIDDLE
from stereo.dataclass import Stereo3DSeries
from aux_funcs.calculations_for_parsivel_data import volume_drop
from stereo.convert_to_parsivel import find_diameter_class

"""
Compute the rain rate in a time series
"""


def rain_rate(
    series: Stereo3DSeries, interval_seconds: int | float
) -> ndarray[float, Any]:
    # Define the ends of the time series
    start, stop = series.duration
    start = floor(start / interval_seconds) * interval_seconds
    stop = ceil(stop / interval_seconds) * interval_seconds

    # Create an empty object with the slots to fit the data
    size_array = int((stop - start) // interval_seconds)
    rain_rate = zeros(shape=(size_array,), dtype=float)

    # Loop thought every row of data and add the rate until you have a value
    for item in series:
        idx = int((item.timestamp_ms - start) // interval_seconds)
        rain_rate[idx] += volume_drop(item.diameter)

    return rain_rate / series.area_of_study / (interval_seconds / 3600)


def time_series(
    series: Stereo3DSeries, interval_seconds: int | float
) -> ndarray[float, Any]:
    # Define the ends of the time series
    start, stop = series.duration
    start = floor(start / interval_seconds) * interval_seconds
    stop = ceil(stop / interval_seconds) * interval_seconds

    return linspace(
        start, stop, int(ceil((stop - start) // interval_seconds)), endpoint=False
    )


"""
Function for calculating the drop_size distribution graph
"""


def get_kinetic_energy(series: Stereo3DSeries) -> float:
    output = 0
    for drop in series:
        output += (volume_drop(drop.diameter) * 1e-9) * (drop.velocity**2) / 2
    return output


"""
Function for calculating the drop_size distribution graph
"""


def get_ndrops(
    series: Stereo3DSeries, min_diam: float = 0.0, max_diam=27.5
) -> Tuple[ndarray, ndarray]:
    n_bins = 100
    length = (max_diam - min_diam) / n_bins
    diameters = linspace(min_diam, max_diam, n_bins + 1)
    middle_points = array(
        [(diameters[i] + diameters[i + 1]) / 2 for i in range(n_bins)]
    )
    ndrops = zeros(n_bins)
    for row in series:
        idx = int((row.diameter - min_diam) // length)
        if idx < n_bins:
            ndrops[idx] += 1
    return (middle_points, ndrops)


"""
Function for calculating the drop_size distribution graph
"""


def get_ndropsxvelocity(
    series: Stereo3DSeries, min_vel: float = 0.0, max_vel=24.0
) -> Tuple[ndarray, ndarray]:
    n_bins = 100
    length = (max_vel - min_vel) / n_bins
    velocities = linspace(min_vel, max_vel, n_bins + 1)
    middle_points = array(
        [(velocities[i] + velocities[i + 1]) / 2 for i in range(n_bins)]
    )
    ndrops = zeros(n_bins)
    for row in series:
        idx = int((row.velocity - min_vel) // length)
        if idx < n_bins:
            ndrops[idx] += 1
    return (middle_points, ndrops)


"""
Function for calculating the drop_size distribution graph
"""


def get_dsd(
    series: Stereo3DSeries, min_diam: float = 0.0, max_diam=27.5
) -> Tuple[ndarray, ndarray]:
    n_bins = 100
    length = (max_diam - min_diam) / n_bins
    diameters = linspace(min_diam, max_diam, n_bins + 1)
    middle_points = array(
        [(diameters[i] + diameters[i + 1]) / 2 for i in range(n_bins)]
    )
    nd3 = zeros(n_bins)
    for row in series:
        idx = int((row.diameter - min_diam) // length)
        nd3[idx] += row.diameter**3
    return (middle_points, nd3)


"""
Function for counting the number of drops in each diameter class
"""


def get_ndrops_in_diameter_classes(series: Stereo3DSeries, parsivel_class: bool = True):
    if not parsivel_class:
        assert False, "Need to include other classes"
    middle_points = CLASSES_DIAMETER_MIDDLE
    ndrops = zeros(32)
    for row in series:
        idx = find_diameter_class(row.diameter) - 1
        ndrops[idx] += 1
    return (middle_points, ndrops / series.area_of_study)
