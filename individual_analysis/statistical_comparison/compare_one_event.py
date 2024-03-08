from pathlib import Path
from matplotlib.pyplot import figure
from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle

OUTPUTFOLDER = Path(__file__).parent / "output"


def generate_comparison(
    parsivel_event: ParsivelTimeSeries, stereo_event: ParsivelTimeSeries
):
    # Ndrops per Area
    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(1,1,1)
    ax.plot(parsivel_event.get_number_of_drops_series() / parsivel_event.area_of_study, color="orangered", label= "Parsivel")
    ax.plot(stereo_event.get_number_of_drops_series() / stereo_event.area_of_study, color="dodgerblue", label= "3D Stereo")
    ax.set_title("Number of Drops per Area")
    ax.set_ylabel("$n.m^{-2}$", fontdict={"fontsize": 14})
    fig.savefig(OUTPUTFOLDER / "quick_look_ndrops.png")

    # Rain rate
    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(1,1,1)
    ax.plot(parsivel_event.rain_rate(), color="orangered", label= "Parsivel")
    ax.plot(stereo_event.rain_rate(), color="dodgerblue", label= "3D Stereo")
    ax.set_title("Rain Rate")
    ax.set_ylabel("$mm.h^{-1}$", fontdict={"fontsize": 14})
    fig.savefig(OUTPUTFOLDER / "quick_look_rain_rate.png")

    # Cumulative rain Depth
    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(1,1,1)
    ax.plot(parsivel_event.cumulative_depth(), color="orangered", label= "Parsivel")
    ax.plot(stereo_event.cumulative_depth(), color="dodgerblue", label= "3D Stereo")
    ax.set_title("Cumulative Depth")
    ax.set_ylabel("$mm$", fontdict={"fontsize": 14})
    fig.savefig(OUTPUTFOLDER / "quick_look_depth.png")

    # dsd per class
    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(1,1,1)
    x, y = parsivel_event.get_nd3()
    ax.plot(x, y, color="orangered", label= "Parsivel")
    x, y = stereo_event.get_nd3()
    ax.plot(x, y, color="dodgerblue", label= "3D Stereo")
    ax.set_title("Drop size distribution")
    ax.set_ylabel("$N(d).d^{3}$", fontdict={"fontsize": 14})
    ax.set_xlabel("Diameter $(mm)$", fontdict={"fontsize": 14})
    ax.set_xbound(0, 6)
    fig.savefig(OUTPUTFOLDER / "quick_look_dsd.png")

    fig = figure(dpi=300)
    fig.set_size_inches((5, 4))
    fig.set_layout_engine("constrained")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", fontsize=16)
    fig.savefig(OUTPUTFOLDER / "quick_look_labels.png")


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
