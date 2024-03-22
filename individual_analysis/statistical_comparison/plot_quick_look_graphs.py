from pathlib import Path
from matplotlib.pyplot import figure
from numpy import fromiter
from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle
from individual_analysis.analysis_variables import (
    # PARSIVELBASECOLOR,
    # STEREOBASECOLOR,
    AXESTITLESFONTSIZE,
    AXESLABELSFONTSIZE,
    LEGENDFIGURESPECS,
    FIGURESPECS,
)

PARSIVELEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
)
STEREOEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
)

OUTPUTFOLDER = Path(__file__).parent / "output"


def main(parsivel_event: ParsivelTimeSeries, stereo_event: ParsivelTimeSeries):
    # Ndrops per Area
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot(1, 1, 1)

    y_parsivel = (
        parsivel_event.get_number_of_drops_series() / parsivel_event.area_of_study
    )
    y_stereo = stereo_event.get_number_of_drops_series() / stereo_event.area_of_study
    series_minutes = fromiter((i / 2 for i in range(y_parsivel.size)), dtype=float)
    ax.plot(series_minutes, y_parsivel, color="orangered", label="Parsivel")
    ax.plot(series_minutes[:-1], y_stereo, color="dodgerblue", label="3D Stereo")
    # ax.set_title("Number of Drops per Area", fontsize= AXESTITLESFONTSIZE})
    ax.set_ylabel("Nb of drops per area $(m^{-2})$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xlabel("Time $(min)$", fontsize=AXESLABELSFONTSIZE)
    fig.savefig(OUTPUTFOLDER / "quick_look_ndrops.png")

    # Rain rate
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot(1, 1, 1)
    y_parsivel = parsivel_event.rain_rate()
    y_stereo = stereo_event.rain_rate()
    ax.plot(series_minutes, y_parsivel, color="orangered", label="Parsivel")
    ax.plot(series_minutes[:-1], y_stereo, color="dodgerblue", label="3D Stereo")
    # ax.set_title("Rain Rate", fontsize= AXESTITLESFONTSIZE)
    ax.set_ylabel("Rain rate $(mm.h^{-1})$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xlabel("Time $(min)$", fontsize=AXESLABELSFONTSIZE)
    fig.savefig(OUTPUTFOLDER / "quick_look_rain_rate.png")

    # Cumulative rain Depth
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot(1, 1, 1)
    y_parsivel = parsivel_event.cumulative_depth()
    y_stereo = stereo_event.cumulative_depth()
    ax.plot(series_minutes, y_parsivel, color="orangered", label="Parsivel")
    ax.plot(series_minutes[:-1], y_stereo, color="dodgerblue", label="3D Stereo")
    # ax.set_title("Cumulative Depth", fontsize= AXESTITLESFONTSIZE)
    ax.set_ylabel("Cumulative depth $(mm)$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xlabel("Time $(min)$", fontsize=AXESLABELSFONTSIZE)
    fig.savefig(OUTPUTFOLDER / "quick_look_depth.png")

    # dsd per class
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot(1, 1, 1)
    x, y = parsivel_event.get_nd3()
    ax.plot(x, y, color="orangered", label="Parsivel")
    x, y = stereo_event.get_nd3()
    ax.plot(x, y, color="dodgerblue", label="3D Stereo")
    # ax.set_title("Drop size distribution", fontsize= AXESTITLESFONTSIZE)
    ax.set_ylabel("$N(d).d^{3}$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xlabel("Diameter $(mm)$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xbound(0, 6)
    handles, labels = ax.get_legend_handles_labels()
    fig.savefig(OUTPUTFOLDER / "quick_look_dsd.png")

    # Plot the labels
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    ax.legend(handles, labels, **LEGENDFIGURESPECS)
    ax.set_axis_off()
    fig.savefig(OUTPUTFOLDER / "quick_look_labels.png")

    # Print information about the event
    print(f"{parsivel_event.duration_readable}")


if __name__ == "__main__":
    print("Reading the events for Parsivel.")
    stereo_events = [
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(STEREOEVENTSFOLDER.iterdir())
    ]
    print("Running analysis and plotting graphs.")
    parsivel_event = next(
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(PARSIVELEVENTSFOLDER.iterdir())
    )
    stereo_event = next(
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(STEREOEVENTSFOLDER.iterdir())
    )
    print("READ THE DATA FOR BOTH DEVICES")
    main(parsivel_event, stereo_event)
    print("DONE.")
