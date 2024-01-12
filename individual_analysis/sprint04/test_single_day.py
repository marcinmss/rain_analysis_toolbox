"""
This file is for testing the day that Auguste presented to Jerry as an example of a lot of drops in a day
 17 Nov 2023 01:21:59
"""

from pathlib import Path
from stereo3d import stereo3d_read_from_zips
from numpy import array, floor, isclose, unique, allclose

beg = 20231116000000
end = 20231116235959

source_folder = Path("/home/marcio/stage_project/data/Daily_raw_data_3D_stereo/")

data = stereo3d_read_from_zips(beg, end, source_folder)

# Lets create an array that contains the time stamps in the rain rate series when there is rain (> 0.0)
resol = 0.001
time_series = data.time_series(resol)
tstamps_where_is_rain01 = array(
    [
        tstep
        for (tstep, is_rain) in zip(time_series, data.rain_rate(resol) > 0.0)
        if is_rain
    ]
)

# Now lets check agains the array of the timestamps of the drops
tstamps_where_is_rain02 = unique([drop.timestamp_ms for drop in data])

tstamps_where_is_rain01[0]
tstamps_where_is_rain02[0]
