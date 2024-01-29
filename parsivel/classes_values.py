"""
This file contains the classes for the standard parsivel matrix
"""

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

CLASSES_VELOCITY_BINS = []
CLASSES_VELOCITY_MIDDLE = []
CLASSES_VELOCITY_MAP = []
left = 0
right = 0
for i, bin_diam in enumerate((item[1] for item in CLASSES_VELOCITY)):
    right += bin_diam
    CLASSES_VELOCITY_BINS.append((left, right))
    CLASSES_VELOCITY_MIDDLE.append((left + right) / 2)
    left = right
    for _ in range(int(bin_diam // 0.1)):
        CLASSES_VELOCITY_MAP.append(i)


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

CLASSES_DIAMETER_BINS = []
CLASSES_DIAMETER_MIDDLE = []
CLASSES_DIAMETER_MAP = []
left = 0
right = 0
for i, bin_diam in enumerate((item[1] for item in CLASSES_DIAMETER)):
    right += bin_diam
    CLASSES_DIAMETER_BINS.append((left, right))
    CLASSES_DIAMETER_MIDDLE.append((left + right) / 2)
    left = right
    for _ in range(int(bin_diam // 0.125)):
        CLASSES_DIAMETER_MAP.append(i)
