from pathlib import Path
from matplotlib.pyplot import figure
from numpy import array, fromiter
from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle

OUTPUTFOLDER = Path(__file__).parent / "output"
AXESTITLESFONTSIZE = 18
AXESLABELSFONTSIZE = 16


def generate_comparison(
    parsivel_event: ParsivelTimeSeries, stereo_event: ParsivelTimeSeries
):
    # Ndrops per Area
    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(1, 1, 1)

    y_parsivel = (
        parsivel_event.get_number_of_drops_series() / parsivel_event.area_of_study
    )
    y_stereo = stereo_event.get_number_of_drops_series() / stereo_event.area_of_study
    series_minutes = fromiter((i / 2 for i in range(y_parsivel.size)), dtype=float)
    ax.plot(series_minutes, y_parsivel, color="orangered", label="Parsivel")
    ax.plot(series_minutes[:-1], y_stereo, color="dodgerblue", label="3D Stereo")
    # ax.set_title("Number of Drops per Area", fontdict={"fontsize": AXESTITLESFONTSIZE})
    ax.set_ylabel(
        "Nb of drops per Area $(m^{-2})$", fontdict={"fontsize": AXESLABELSFONTSIZE}
    )
    ax.set_xlabel("Time $(min)$", fontdict={"fontsize": AXESLABELSFONTSIZE})
    fig.savefig(OUTPUTFOLDER / "quick_look_ndrops.png")

    # Rain rate
    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(1, 1, 1)
    y_parsivel = parsivel_event.rain_rate()
    y_stereo = stereo_event.rain_rate()
    ax.plot(series_minutes, y_parsivel, color="orangered", label="Parsivel")
    ax.plot(series_minutes[:-1], y_stereo, color="dodgerblue", label="3D Stereo")
    # ax.set_title("Rain Rate", fontdict={"fontsize": AXESTITLESFONTSIZE})
    ax.set_ylabel("Rain Rate $(mm.h^{-1})$", fontdict={"fontsize": AXESLABELSFONTSIZE})
    ax.set_xlabel("Time $(min)$", fontdict={"fontsize": AXESLABELSFONTSIZE})
    fig.savefig(OUTPUTFOLDER / "quick_look_rain_rate.png")

    # Cumulative rain Depth
    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(1, 1, 1)
    y_parsivel = parsivel_event.cumulative_depth()
    y_stereo = stereo_event.cumulative_depth()
    ax.plot(series_minutes, y_parsivel, color="orangered", label="Parsivel")
    ax.plot(series_minutes[:-1], y_stereo, color="dodgerblue", label="3D Stereo")
    # ax.set_title("Cumulative Depth", fontdict={"fontsize": AXESTITLESFONTSIZE})
    ax.set_ylabel("Cumulative Depth $(mm)$", fontdict={"fontsize": AXESLABELSFONTSIZE})
    ax.set_xlabel("Time $(min)$", fontdict={"fontsize": AXESLABELSFONTSIZE})
    fig.savefig(OUTPUTFOLDER / "quick_look_depth.png")

    # dsd per class
    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(1, 1, 1)
    x, y = parsivel_event.get_nd3()
    ax.plot(x, y, color="orangered", label="Parsivel")
    x, y = stereo_event.get_nd3()
    ax.plot(x, y, color="dodgerblue", label="3D Stereo")
    # ax.set_title("Drop size distribution", fontdict={"fontsize": AXESTITLESFONTSIZE})
    ax.set_ylabel("$N(d).d^{3}$", fontdict={"fontsize": AXESLABELSFONTSIZE})
    ax.set_xlabel("Diameter $(mm)$", fontdict={"fontsize": AXESLABELSFONTSIZE})
    ax.set_xbound(0, 6)
    handles, labels = ax.get_legend_handles_labels()
    fig.savefig(OUTPUTFOLDER / "quick_look_dsd.png")

    # Plot the labels
    fig = figure(dpi=300, figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.legend(handles, labels, loc="center", fontsize=AXESLABELSFONTSIZE, frameon=False)
    ax.set_axis_off()
    fig.savefig(OUTPUTFOLDER / "quick_look_labels.png")

    # Print information about the event
    print(f"{parsivel_event.duration_readable}")


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
