import matplotlib
from pathlib import Path
from sprint06.gamma_dist import nd_parsivel
from parsivel import pars_read_from_pickle
from matplotlib import pyplot as plt

parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/parsivel/"
)
stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/stereo/"
)

parsivel_events = [
    pars_read_from_pickle(file) for file in parsivel_events_folder.iterdir()
]

matrix = parsivel_events[0].matrix_for_event


plt.plot(nd_parsivel(matrix))
plt.show()
