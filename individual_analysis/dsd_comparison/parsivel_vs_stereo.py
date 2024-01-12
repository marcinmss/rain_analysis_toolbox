from pathlib import Path
from matplotlib.pyplot import figure
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
output_folder = Path(__file__).parent / "output/"

if __name__ == "__main__":
    # Read the data for both devices
    stereo_event = stereo_read_from_pickle(
        next(stereo_events_folder.iterdir())
    ).convert_to_parsivel()
    parsivel_event = pars_read_from_pickle(next(parsivel_events_folder.iterdir()))

    # Plot the data and save figure
    fig = figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_dsd(parsivel_event, ax)
    plot_dsd(stereo_event, ax, BASESTEREOSTYLE)
    ax.set_xbound(0, 3)
    ax.legend(prop={"size": 6}, framealpha=0.0, loc="best")
    fig.savefig(output_folder / "dsd_analysis_parsivel_vs_stereo.png")
