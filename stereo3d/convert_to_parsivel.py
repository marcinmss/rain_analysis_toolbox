from parsivel import ParsivelTimeSeries, ParsivelTimeStep, AREAPARSIVEL
from stereo3d import Stereo3DSeries, BASEAREASTEREO3D
from aux_funcs.aux_datetime import tstamp_to_dt, range_dtime_30s, dt_to_tstamp
from aux_funcs.bin_data import bin_diameter, bin_velocity

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

    print(new_series)
    t0 = (new_series[0]).timestamp

    factor = AREAPARSIVEL / BASEAREASTEREO3D
    # factor = 1
    for item in series:
        # generate the matrix for the
        idx = int((item.timestamp - t0) // 30)
        class_velocity = bin_velocity(item.velocity)
        class_diameter = bin_diameter(item.diameter)

        if 1 <= class_velocity <= 32 and 0 < class_diameter < 33:
            new_series[idx].matrix[class_diameter - 1, class_velocity - 1] += factor

    return ParsivelTimeSeries("3D Stereo", series.duration, [], new_series, 30)
