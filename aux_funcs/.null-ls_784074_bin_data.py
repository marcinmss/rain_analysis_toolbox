from parsivel.matrix_classes import 
"""
This modules has the classes for putting continuous data in each class for the
Parcivel device.
"""

TIMEBOX = 30


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


def find_class_midle_point(class_number: int) -> int:
    assert 1 < class_number < 32
    return CLASSES_DIAMETER_MIDDLE[class_number - 1]


def bin_diameter(diameter: float) -> int:
    assert diameter > 0, f"Impossible value: {diameter}"
    for id, (center, width) in enumerate(CLASSES_DIAMETER, start=1):
        if center - width / 2 < diameter < center + width / 2:
            return id

    return 33



def find_velocity_class(diameter: float) -> int:
    assert diameter > 0, f"Impossible value: {diameter}"
    idx = int(diameter // 0.1)
    if idx > 31:
        return 32
    else:
        return CLASSES_VELOCITY_MAP[idx] + 1


def find_velocity_mp(class_number: int) -> int:
    assert 1 < class_number < 32
    return CLASSES_VELOCITY_MIDDLE[class_number - 1]


def bin_velocity(velocity: float) -> int:
    [s, d] = CLASSES_VELOCITY[0]
    if velocity < s - d / 2:
        return 0
    for i, [_, d] in enumerate(CLASSES_VELOCITY):
        if velocity < s + d:
            return i + 1
        else:
            s = s + d
    return 33
