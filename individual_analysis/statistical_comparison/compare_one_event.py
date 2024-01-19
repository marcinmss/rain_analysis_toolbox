from pathlib import Path
from numpy import ceil
from matplotlib.pyplot import figure
from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle
from stereo.dataclass import BASESTEREOSTYLE
from parsivel.dataclass import BASEPARSIVELSTYLE as BASEPARSIVELSTYLE

OUTPUTFOLDER = Path(__file__).parent / "output"


def generate_comparison(
    parsivel_event: ParsivelTimeSeries, stereo_event: ParsivelTimeSeries
):
    number_indicators = 5
    ncols = 3
    nrows = int(ceil(number_indicators / ncols))
    counter = 1

    fig = figure()
    # figure.suptitle(title, fontsize= 16)
    fig.set_size_inches(16, nrows * 4)
    fig.set_layout_engine("constrained")

    # N(d) series
    ax = fig.add_subplot(nrows, ncols, counter)
    ax.plot(parsivel_event.number_drops_per_area_series(), **BASEPARSIVELSTYLE)
    ax.plot(stereo_event.number_drops_per_area_series(), **BASESTEREOSTYLE)
    ax.set_title("Number of Drops per Area")
    ax.set_ylabel("$n.m^{-2}$")
    counter += 1

    # Rain rate
    ax = fig.add_subplot(nrows, ncols, counter)
    ax.plot(parsivel_event.rain_rate(), **BASESTEREOSTYLE)
    ax.plot(stereo_event.rain_rate(), **BASEPARSIVELSTYLE)
    ax.set_title("Rain Rate")
    ax.set_ylabel("$mm.h^{-1}$")
    counter += 1

    # Cumulative rain Depth
    ax = fig.add_subplot(nrows, ncols, counter)
    ax.plot(parsivel_event.cumulative_depth(), **BASEPARSIVELSTYLE)
    ax.plot(stereo_event.cumulative_depth(), **BASESTEREOSTYLE)
    ax.set_title("Cumulative Depth")
    ax.set_ylabel("$mm$")
    counter += 1

    # Ndrops per Class
    ax = fig.add_subplot(nrows, ncols, counter)
    x, y = parsivel_event.get_nd()
    ax.plot(x, y, **BASEPARSIVELSTYLE)
    x, y = stereo_event.get_nd()
    ax.plot(x, y, **BASESTEREOSTYLE)
    ax.set_title("Number of drops divided by Area for each Class")
    ax.set_ylabel("N(d)")
    ax.set_xlabel("Diameter $(mm)$")
    ax.set_xbound(0, 6)
    counter += 1

    # dsd per class
    ax = fig.add_subplot(nrows, ncols, counter)
    x, y = parsivel_event.get_nd3()
    ax.plot(x, y, **BASEPARSIVELSTYLE)
    x, y = stereo_event.get_nd3()
    ax.plot(x, y, **BASESTEREOSTYLE)
    ax.set_title("Drop size distribution")
    ax.set_ylabel("$N(d).d^{3}$")
    ax.set_xlabel("Diameter $(mm)$")
    ax.set_xbound(0, 6)
    counter += 1

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.75, 0.25), fontsize=16)
    fig.savefig(OUTPUTFOLDER / "statistical_comparison_one_event.png")


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
