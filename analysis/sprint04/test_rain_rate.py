from pathlib import Path

from numpy import allclose, array, unique, where
from stereo3d import stereo_read_from_pickle
from matplotlib.pyplot import figure

stereo_events_folder = Path(
    "/home/marcio/stage_project/individual_analysis/sprint02/saved_events/stereo"
)

# First lets plot the rain rate for an expecifyc event and them
stereo_example = stereo_read_from_pickle(next(stereo_events_folder.iterdir()))


# Now lets create an array that is only true when there is rain rate
ocurrency_array01 = stereo_example.rain_rate(0.001) > 0.0
time_series = stereo_example.time_series(0.001)
tstamps_where_is_rain01 = array(
    [tstep for (tstep, is_rain) in zip(time_series, ocurrency_array01) if is_rain]
)

# Now to Check if every thing is correct we can get
tstamps_where_is_rain02 = unique([drop.timestamp_ms for drop in stereo_example])

allclose(tstamps_where_is_rain01, tstamps_where_is_rain02)

tstamps_where_is_rain01 - tstamps_where_is_rain02
