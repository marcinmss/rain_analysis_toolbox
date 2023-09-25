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

CLASSES_DIAMETER = [
    [0.062, 0.125],
    [0.187, 0.125],
    [0.312, 0.125],
    [0.437, 0.125],
    [0.562, 0.125],
    [0.687, 0.125],
    [0.812, 0.125],
    [0.937, 0.125],
    [1.062, 0.125],
    [1.187, 0.125],
    [1.375, 0.25],
    [1.625, 0.25],
    [1.875, 0.25],
    [2.125, 0.25],
    [2.375, 0.25],
    [2.75, 0.5],
    [3.25, 0.5],
    [3.75, 0.5],
    [4.25, 0.5],
    [4.75, 0.5],
    [5.5, 1.0],
    [6.5, 1.0],
    [7.5, 1.0],
    [8.5, 1.0],
    [9.5, 1.0],
    [11.0, 2.0],
    [13.0, 2.0],
    [15.0, 2.0],
    [17.0, 2.0],
    [19.0, 2.0],
    [21.5, 3.0],
    [24.5, 3.0],
]


def bin_diameter(diameter: float) -> int:
    assert diameter > 0, f"Impossible value: {diameter}"
    for id, (center, width) in enumerate(CLASSES_DIAMETER, start=1):
        if center - width / 2 < diameter < center + width / 2:
            return id

    return 33


CLASSES_VELOCITY = [
    [0.05, 0.1],
    [0.15, 0.1],
    [0.25, 0.1],
    [0.35, 0.1],
    [0.45, 0.1],
    [0.55, 0.1],
    [0.65, 0.1],
    [0.75, 0.1],
    [0.85, 0.1],
    [0.95, 0.1],
    [1.1, 0.2],
    [1.3, 0.2],
    [1.5, 0.2],
    [1.7, 0.2],
    [1.9, 0.2],
    [2.2, 0.4],
    [2.6, 0.4],
    [3.0, 0.4],
    [3.4, 0.4],
    [3.8, 0.4],
    [4.4, 0.8],
    [5.2, 0.8],
    [6.0, 0.8],
    [6.8, 0.8],
    [7.6, 0.8],
    [8.8, 1.6],
    [10.4, 1.6],
    [12.0, 1.6],
    [13.6, 1.6],
    [15.2, 1.6],
    [17.6, 3.2],
    [20.8, 3.2],
]


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
