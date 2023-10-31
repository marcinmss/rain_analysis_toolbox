from typing import Literal
from numpy import ndarray, sum
from aux_funcs.general import volume_drop
from parsivel.matrix_classes import CLASSES_DIAMETER_MIDDLE

"""
Given a standard 32x32 Parsivel matrix, calculates the aproximate total volume
using the equivalent diameter
"""


def matrix_to_volume(matrix: ndarray | Literal[0]) -> float:
    n_per_diameter = sum(matrix, axis=0)
    return sum(
        [n * volume_drop(d) for (n, d) in zip(n_per_diameter, CLASSES_DIAMETER_MIDDLE)]
    )
