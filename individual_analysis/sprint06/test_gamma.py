from pathlib import Path
from sprint06.gamma_dist import gamma_dist, nd_parsivel, dsd_params
from parsivel import pars_read_from_pickle
from matplotlib import pyplot as plt

parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/parsivel/"
)
stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/stereo/"
)

parsivel_event = pars_read_from_pickle(next(parsivel_events_folder.iterdir()))

matrix = parsivel_event.matrix_for_event

nd = nd_parsivel(matrix)
params = dsd_params(matrix)
gamma = gamma_dist(params.lamb, params.mu, params.n0)
figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
ax.plot(nd, c="orangered", label="gamma")
ax.plot(gamma, c="dodgerblue", label="nd")
figure.show()
