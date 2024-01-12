from pathlib import Path
from matplotlib.pyplot import figure, show
from stereo3d import stereo_read_from_pickle

from stereo3d.plots import BASESTEREOSTYLE
from parsivel import pars_read_from_pickle
from parsivel.plots import plot_dsd

parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/parsivel/"
)

stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/stereo/"
)

stereo_event = (
    stereo_read_from_pickle(next(stereo_events_folder.iterdir()))
    .convert_to_parsivel()
    .filter_by_parsivel_resolution()
)
parsivel_event = pars_read_from_pickle(next(parsivel_events_folder.iterdir()))


fig = figure()
ax = fig.add_subplot(1, 1, 1)
plot_dsd(parsivel_event, ax)
plot_dsd(stereo_event, ax, BASESTEREOSTYLE)
show()
