from pathlib import Path
from matplotlib.pyplot import figure
from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle

OUTPUTFOLDER = Path(__file__).parent / "output"


def generate_comparison(
    parsivel_event: ParsivelTimeSeries, stereo_event: ParsivelTimeSeries
):
    ncols = 4
    nrows = 1
    counter = 1

    fig = figure()
    # figure.suptitle(title, fontsize= 16)
    fig.set_size_inches(16, nrows * 4)
    fig.set_layout_engine("constrained")

    # N(d) series
    ax = fig.add_subplot(nrows, ncols, counter)
    ax.plot(
        parsivel_event.number_drops_per_area_series(),
        color="orangered",
        label="Parsivel",
    )
    ax.plot(
        stereo_event.number_drops_per_area_series(), "dodgerblue", label="3D Stereo"
    )
    ax.set_title("Number of Drops per Area")
    ax.set_ylabel("$n.m^{-2}$")
    counter += 1

    # Rain rate
    ax = fig.add_subplot(nrows, ncols, counter)
    ax.plot(parsivel_event.rain_rate(), "dodgerblue", label="3D Stereo")
    ax.plot(stereo_event.rain_rate(), color="orangered", label="Parsivel")
    ax.set_title("Rain Rate")
    ax.set_ylabel("$mm.h^{-1}$")
    counter += 1

    # Cumulative rain Depth
    ax = fig.add_subplot(nrows, ncols, counter)
    ax.plot(parsivel_event.cumulative_depth(), color="orangered", label="Parsivel")
    ax.plot(stereo_event.cumulative_depth(), "dodgerblue", label="3D Stereo")
    ax.set_title("Cumulative Depth")
    ax.set_ylabel("$mm$")
    counter += 1

    # dsd per class
    ax = fig.add_subplot(nrows, ncols, counter)
    x, y = parsivel_event.get_nd3()
    ax.plot(x, y, color="orangered", label="Parsivel")
    x, y = stereo_event.get_nd3()
    ax.plot(x, y, "dodgerblue", label="3D Stereo")
    ax.set_title("Drop size distribution")
    ax.set_ylabel("$N(d).d^{3}$")
    ax.set_xlabel("Diameter $(mm)$")
    ax.set_xbound(0, 6)
    counter += 1

    ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc=(1.15, 0.25), fontsize=16)
    fig.savefig(OUTPUTFOLDER / "quick_look.png")


if __name__ == "__main__":
    pars_folder = Path(
        "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
    )
    stereo_folder = Path(
        "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
    )
    parsivel_event = next(
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(pars_folder.iterdir())
    )
    stereo_event = next(
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(stereo_folder.iterdir())
    )
    print("READ THE DATA FOR BOTH DEVICES")
    generate_comparison(parsivel_event, stereo_event)
    print("DONE.")
