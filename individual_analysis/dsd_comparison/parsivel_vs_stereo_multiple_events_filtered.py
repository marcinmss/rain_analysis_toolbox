from pathlib import Path
from matplotlib.pyplot import figure
from stereo3d import stereo_read_from_pickle

from stereo3d.plots import BASESTEREOSTYLE
from parsivel import parsivel_read_from_pickle
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
    n_events = 16
    stereo_events = [
        stereo_read_from_pickle(file)
        .convert_to_parsivel()
        .filter_by_parsivel_resolution()
        for i, file in enumerate(stereo_events_folder.iterdir())
        if i < n_events
    ]
    parsivel_events = [
        parsivel_read_from_pickle(file)
        for i, file in enumerate(parsivel_events_folder.iterdir())
        if i < n_events
    ]

    # Plot the data and save figure
    fig = figure()
    fig.set_layout_engine("constrained")
    fig.set_size_inches(20, 20)
    for i, (parsivel_event, stereo_event) in enumerate(
        zip(parsivel_events, stereo_events)
    ):
        ax = fig.add_subplot(4, 4, i + 1)
        plot_dsd(parsivel_event, ax)
        plot_dsd(stereo_event, ax, BASESTEREOSTYLE)
        ax.legend(prop={"size": 6}, framealpha=0.0, loc="upper left")
        ax.set_xbound(0, 3)
    fig.savefig(
        output_folder / "dsd_analysis_multiple_events_parsivel_vs_stereo_filtered.png"
    )
