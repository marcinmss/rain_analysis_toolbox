from parsivel import ParsivelTimeSeries, ParsivelTimeStep
from stereo3d import Stereo3DSeries
from aux_funcs.aux_datetime import tstamp_to_dt, range_dtime_30s, dt_to_tstamp
from parsivel.matrix_classes import CLASSES_DIAMETER_BINS, CLASSES_VELOCITY_BINS

"""
Auxiliary functions for classifiing the diameter and the speed into the Parsivel
standard classifications.
It provides us with 32 bins (1 to 32) if the diameter its outside to the right
it will be classified with 33 and if its outside to the left 0.
"""


def find_diameter_class(diameter: float) -> int:
    assert diameter > 0, f"Impossible value: {diameter}"
    for i, (left, right) in enumerate(CLASSES_DIAMETER_BINS):
        if left <= diameter < right:
            return i + 1
    return 32


def find_velocity_class(velocity: float) -> int:
    assert velocity > 0, f"Impossible value: {velocity}"
    for i, (beg, end) in enumerate(CLASSES_VELOCITY_BINS):
        if beg <= velocity < end:
            return i + 1
    return 32


"""
Converts the data from the 3D stereo to the parsivel format arranging the 
drops in the standard matrix.
"""


def convert_to_parsivel(series: Stereo3DSeries) -> ParsivelTimeSeries:
    dtime0, dtimef = tstamp_to_dt(series.duration[0]), tstamp_to_dt(series.duration[1])

    new_series = [
        ParsivelTimeStep.zero_like(dt_to_tstamp(dtime))
        for dtime in range_dtime_30s(dtime0, dtimef)
    ]

    t0 = series.duration[0]

    for item in series:
        idx = int((item.timestamp - t0) // 30)
        class_velocity = find_velocity_class(item.velocity)
        class_diameter = find_diameter_class(item.diameter)

        if 1 <= class_velocity <= 32 and 0 < class_diameter < 33:
            new_series[idx].matrix[class_velocity - 1, class_diameter - 1] += 1

    return ParsivelTimeSeries(
        "stereo3d", series.duration, [], new_series, 30, series.area_of_study
    )
