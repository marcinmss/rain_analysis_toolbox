from numpy import ndarray, sum, pi, zeros
from .bin_data import CLASSES_DIAMETER
from typing import List

"""
Given the standard 32x32 matrix from the parsivel, calculate the hourly rain
rate.
"""
AREAPARSIVEL = 5400


"""
Volume of an spherical drop
"""


def volume_drop(diameter: float) -> float:
    return diameter**3 * pi / 6


"""
Function for obtaining the equivalent diameter of an class, found using the 
mean value theorem.
"""


def equivalent_diameter(center: float, width: float) -> float:
    d0, df = center - width * 0.5, center + width * 0.5
    return ((df**3 + df**2 * d0 + df * d0**2 + d0**3) * 0.25) ** (1 / 3)


VOLUME_PER_CLASS_ED = [
    volume_drop(equivalent_diameter(c, w)) for c, w in CLASSES_DIAMETER
]

"""
Uses the mean diameter for calculating the volume of each class
"""

VOLUME_PER_CLASS_MD = [volume_drop(CLASSES_DIAMETER[i][0]) for i in range(32)]

"""
Given a standard 32x32 Parsivel matrix, calculates the aproximate total volume
using the equivalent diameter
"""


def matrix_to_volume(matrix: ndarray) -> float:
    n_per_diameter = sum(matrix, axis=1)
    return sum([n * v for (n, v) in zip(n_per_diameter, VOLUME_PER_CLASS_ED)])


def matrix_to_volume2(matrix: ndarray) -> float:
    n_per_diameter = sum(matrix, axis=1)
    return sum([n * v for (n, v) in zip(n_per_diameter, VOLUME_PER_CLASS_MD)])


"""
Given a standard 32x32 Parsivel matrix, calculates the aproximate hourly rain
rate.
"""


def matrix_to_rainrate(matrix: ndarray, area: float) -> float:
    return matrix_to_volume(matrix) / (area * (30.0 / 3600.0))


def matrix_to_rainrate2(matrix: ndarray, area: float) -> float:
    return matrix_to_volume2(matrix) / (area * (30.0 / 3600.0))


"""
Aggregate a list of parsivel data into one item.
"""


# def agregate_data(data: List[ParsivelInfo]) -> ParsivelInfo:
#     n = len(data)
#     timestamp = data[0].timestamp
#     temp = 0.0
#     matrix = zeros((32, 32))
#     rate = 0.0
#     for item in data:
#         temp += item.temperature / n
#         matrix += item.matrix
#         rate += item.rain_rate
#
#     return ParsivelInfo(timestamp, rate, temp, matrix)
