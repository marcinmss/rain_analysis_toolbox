from pathlib import Path
from typing import List
from matplotlib.pyplot import figure
from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle
from individual_analysis.analysis_variables import (
    LEGENDSPECTS,
    PARSIVELBASECOLOR,
    STEREOBASECOLOR,
    AXESLABELSFONTSIZE,
    FIGURESPECS,
)

OUTPUTFOLDER = Path(__file__).parent / "output"
PARSIVELEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
)
STEREOEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
)

"""
Plot the Drop size distribution graph using all the events for both devices
"""


def main(
    parsivel_events: List[ParsivelTimeSeries],
    stereo_events: List[ParsivelTimeSeries],
):

    # Get the N.d^3 values for parsivel and 3D Stereo
    x, _ = parsivel_events[0].get_nd3()
    y_parsivel = sum(event.get_nd3()[1] for event in parsivel_events) / len(
        parsivel_events
    )
    y_stereo = sum(event.get_nd3()[1] for event in stereo_events) / len(stereo_events)

    # Define the figure and plot the values
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_parsivel, color=PARSIVELBASECOLOR, label="Parsivel")
    ax.plot(x, y_stereo, color=STEREOBASECOLOR, label="3D Stereo")

    # Custumize the Axis
    ax.set_ylabel("$N(d).d^{3}$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xlabel("Diameter $(mm)$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xbound(0, 6)
    ax.legend(**LEGENDSPECTS)

    # Save figure
    fig.savefig(OUTPUTFOLDER / "dsd_comparison_with_all_events.png")


if __name__ == "__main__":
    print("Reading the events for Parsivel.")
    parsivel_events = [
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(PARSIVELEVENTSFOLDER.iterdir())
    ]
    print("Reading the events for 3D Stereo.")
    stereo_events = [
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(STEREOEVENTSFOLDER.iterdir())
    ]
    print("READ THE DATA FOR BOTH DEVICES")
    print("Running analysis and plotting graphs.")
    main(parsivel_events, stereo_events)
    print("DONE.")
