from numpy import array
from parsivel.parsivel_dataclass import ParsivelTimeSeries
from parsivel.matrix_classes import CLASSES_DIAMETER_BINS
from stereo.dataclass import Stereo3DSeries


"""
Filter a parsivel Series to not include drops outside the parsivel minimal resolution
"""


def parsivel_filter(series: Stereo3DSeries) -> Stereo3DSeries:
    left = CLASSES_DIAMETER_BINS[1][1]
    new_rows = [row for row in series if left < row.diameter]

    return Stereo3DSeries(
        series.device, series.duration, array(new_rows), series.limits_area_of_study
    )


"""
Filter by the proximity to the Hermite tendency line
"""


def hermite_filter(series: ParsivelTimeSeries) -> None:
    pass
