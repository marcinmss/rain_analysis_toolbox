from typing import Tuple
from numpy import array, linspace, ndarray, zeros
from stereo3d.stereo3d_dataclass import Stereo3DSeries

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
