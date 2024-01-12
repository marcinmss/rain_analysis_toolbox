from pathlib import Path
from matplotlib.pyplot import figure
from matplotlib import colormaps

from stereo3d import stereo_read_from_pickle

from parsivel.plots import plot_dsd

stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/stereo/"
)
output_folder = Path(__file__).parent / "output/"

if __name__ == "__main__":
    # Read the data for both devices
    stereo_event = stereo_read_from_pickle(next(stereo_events_folder.iterdir()))
    events_sections = (
        event.convert_to_parsivel()
        for event in stereo_event.split_by_distance_to_sensor()
    )

    cmap = colormaps["autumn_r"]
    fig = figure()
    ax = fig.add_subplot(1, 1, 1)
    for nsection, section in enumerate(events_sections, 1):
        style = {"label": f"section_{nsection}", "color": cmap((nsection + 1) / 9)}
        plot_dsd(section, ax, style)

    ax.set_xbound(0, 3)
    ax.legend(prop={"size": 6}, framealpha=0.0, loc="best")
    # Plot the data and save figure
    fig.savefig(output_folder / "dsd_analysis_stereo_sections.png")
