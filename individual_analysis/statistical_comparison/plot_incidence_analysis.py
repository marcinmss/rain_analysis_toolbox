from parsivel import parsivel_read_from_pickle
from matplotlib.pyplot import figure
from pathlib import Path
from numpy import linspace, zeros
from individual_analysis.analysis_variables import (
    AXESLABELSFONTSIZE,
    PARSIVELBASECOLOR,
    STEREOBASECOLOR,
    FIGURESPECS,
)

PARSIVELEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
)
STEREOEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
)

OUTPUTFOLDER = Path(__file__).parent / "output"


def rain_rate(volume: float, area: float, resolution_seconds: float):
    return volume / area / resolution_seconds


if __name__ == "__main__":

    print("Reading the events for Parsivel.")
    parsivel_events = [
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(PARSIVELEVENTSFOLDER.iterdir())
    ]
    print("Reading the events for 3D Stereo.")
    stereo_converted_events = [
        parsivel_read_from_pickle(file_path)
        for file_path in sorted(STEREOEVENTSFOLDER.iterdir())
    ]
    print("READ THE DATA FOR BOTH DEVICES")

    print("Running analysis and plotting graphs.")
    # Define the size of the classes that will be used in the x axis
    LEFT = 0
    RIGHT = 100
    NBINS = 20
    BINSIZE = (RIGHT - LEFT) / NBINS
    x_limts = linspace(LEFT, RIGHT, NBINS + 1)
    x = (x_limts[:-1] + x_limts[1:]) / 2

    # Loop though all the time steps and add the depth to each class
    parsivel_time_steps = (tstep for event in parsivel_events for tstep in event)
    stereo_time_steps = (tstep for event in stereo_converted_events for tstep in event)
    y_stereo, y_parsivel = zeros(x.shape), zeros(x.shape)
    area_stereo = stereo_converted_events[0].area_of_study
    area_parsivel = parsivel_events[0].area_of_study
    for stereo_tstep, parsivel_tstep in zip(stereo_time_steps, parsivel_time_steps):
        stereo_rain_rate = stereo_tstep.volume_mm3 / area_stereo * 120
        y_stereo[int((stereo_rain_rate - LEFT) // BINSIZE)] += stereo_rain_rate / 120
        parsivel_rain_rate = parsivel_tstep.volume_mm3 / area_parsivel * 120
        y_parsivel[int((parsivel_rain_rate - LEFT) // BINSIZE)] += (
            parsivel_rain_rate / 120
        )

    y_error = (y_parsivel - y_stereo) / y_parsivel

    # Plot the Depth vs Rain Rate graph
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    ax.plot(x, y_stereo, c=STEREOBASECOLOR, label="3D Stereo")
    print(f"max_y_stereo = {max(y_stereo)}")
    ax.plot(x, y_parsivel, c=PARSIVELBASECOLOR, label="Parsivel")
    ax.set_xlabel("Rain rate $(mm.h^{-1})$", fontsize=AXESLABELSFONTSIZE)
    ax.set_ylabel("Cummulative depth $(mm)$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xbound(-0.5, 80)
    ax.legend()
    fig.savefig(OUTPUTFOLDER / "depth_vs_rain_rate.png")

    # Plot the Error from the first graph
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    ax.plot(x, y_error, color=STEREOBASECOLOR)
    ax.set_xbound(-0.5, 80)
    ax.set_xlabel("Rain rate $(mm.h^{-1})$", fontsize=AXESLABELSFONTSIZE)
    ax.set_ylabel("Error", fontsize=AXESLABELSFONTSIZE)
    fig.savefig(OUTPUTFOLDER / "error_vs_rain_rate.png")
    print("DONE.")
