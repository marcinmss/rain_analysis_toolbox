from pathlib import Path
from matplotlib.pyplot import figure
from matplotlib import colormaps

from stereo import stereo_read_from_pickle

# from parsivel.plots import plot_dsd

stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/stereo_full_event.obj"
)
output_folder = Path(__file__).parent / "output/"

if __name__ == "__main__":
    # Read the data for both devices
    stereo_event = stereo_read_from_pickle(stereo_events_folder)
    print("Read the full event stereo.")
    sections = stereo_event.split_by_distance_to_sensor()
    del stereo_event
    print("Split the full event in sections")

    cmap = colormaps["autumn_r"]
    fig = figure()
    fig.set_dpi(300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel("$N(d).d^3$", fontdict={"fontsize": 13})
    ax.set_xlabel("$diameter (mm)$", fontdict={"fontsize": 13})
    for nsection, section in enumerate(sections, 1):
        converted_section = section.convert_to_parsivel()
        style = {"label": f"section_{nsection}", "color": cmap((nsection + 1) / 10)}
        x, y = converted_section.get_nd3()
        ax.plot(x, y, color="black", linewidth=2.1)
        ax.plot(x, y, **style)

    ax.set_xbound(0, 3)
    ax.legend(prop={"size": 6}, framealpha=0.0, loc="best")
    # Plot the data and save figure
    fig.savefig(output_folder / "dsd_analysis_stereo_sections.png")
    print("Done.")
