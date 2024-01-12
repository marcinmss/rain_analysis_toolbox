from pathlib import Path
from matplotlib.pyplot import figure
from stereo3d import stereo_read_from_pickle

from parsivel.plots import plot_rain_rate
from stereo3d.plots import plot_rain_rate as plot_rain_rate2

stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/stereo/"
)

stereo_event = stereo_read_from_pickle(next(stereo_events_folder.iterdir()))
stereo_converted_events = stereo_event.convert_to_parsivel()


fig = figure()
fig.set_size_inches(20, 5)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
plot_rain_rate(stereo_converted_events, ax1)
plot_rain_rate2(stereo_event, ax2)
fig.show()
