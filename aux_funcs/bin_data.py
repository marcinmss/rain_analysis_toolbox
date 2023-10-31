from parsivel.matrix_classes import (
    CLASSES_DIAMETER_MAP,
    CLASSES_VELOCITY_MAP,
)

"""
Auxiliary functions for classifiing the diameter and the speed into the Parsivel
standard classifications.
It provides us with 32 bins (1 to 32) if the diameter its outside to the right
it will be classified with 33 and if its outside to the left 0.
"""


def find_diameter_class(diameter: float) -> int:
    assert diameter > 0, f"Impossible value: {diameter}"
    idx = int(diameter // 0.125)
    if idx > 31:
        return 32
    else:
        return CLASSES_DIAMETER_MAP[idx] + 1


def find_velocity_class(diameter: float) -> int:
    assert diameter > 0, f"Impossible value: {diameter}"
    idx = int(diameter // 0.1)
    if idx > 31:
        return 32
    else:
        return CLASSES_VELOCITY_MAP[idx] + 1
