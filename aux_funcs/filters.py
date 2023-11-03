from parsivel.parsivel_dataclass import ParsivelTimeSeries


"""
Filter a parsivel Series to not include drops outside the parsivel minimal resolution
"""


def resolution_filter(series: ParsivelTimeSeries) -> None:
    for tstep in series:
        for class_diameter in (1, 2):
            for class_velocity in range(1, 33):
                tstep.matrix[class_velocity - 1, class_diameter - 1] = 0


"""
Filter by the proximity to the Hermite tendency line
"""


def hermite_filter(series: ParsivelTimeSeries) -> None:
    pass
