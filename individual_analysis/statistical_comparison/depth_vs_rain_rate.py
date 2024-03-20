from typing import List
from parsivel import parsivel_read_from_pickle as pars_read
from matplotlib.pyplot import figure
from parsivel import ParsivelTimeSeries
from pathlib import Path
from numpy import linspace, zeros

output_folder = Path(__file__).parent / "output"
AXESTITLESFONTSIZE = 18
AXESLABELSFONTSIZE = 16
LEGENDFONTSIZE = 20


def rain_rate(volume: float, area: float, resolution_seconds: float):
    return volume / area / resolution_seconds


def main(
    parsivel_events: List[ParsivelTimeSeries],
    stereo_converted_events: List[ParsivelTimeSeries],
):
    # Create array of all the time steps for both devie's events
    parsivel_time_steps = (tstep for event in parsivel_events for tstep in event)

    stereo_time_steps = (tstep for event in stereo_converted_events for tstep in event)

    LEFT = 0
    RIGHT = 100
    NBINS = 20
    BINSIZE = (RIGHT - LEFT) / NBINS
    y_limts = linspace(LEFT, RIGHT, NBINS + 1)
    x = (y_limts[:-1] + y_limts[1:]) / 2

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

    # Plot the first graph
    fig = figure(dpi=300, figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_stereo, c="dodgerblue", label="3D Stereo")
    print(f"max_y_stereo = {max(y_stereo)}")
    ax.plot(x, y_parsivel, c="orangered", label="Parsivel")
    ax.set_xlabel("Rain Rate $(mm.h^{-1})$", fontsize=AXESLABELSFONTSIZE)
    ax.set_ylabel("Cummulative Depth $(mm)$", fontsize=AXESLABELSFONTSIZE)
    ax.set_xbound(-0.5, 80)
    ax.legend()
    fig.savefig(output_folder / "depth_vs_rain_rate.png")

    # Plot the first graph
    fig = figure(dpi=300, figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_error, c="dodgerblue")
    ax.set_xlabel("Rain Rate $(mm.h^{-1})$", fontsize=AXESLABELSFONTSIZE)
    ax.set_ylabel("Error", fontsize=AXESLABELSFONTSIZE)
    fig.savefig(output_folder / "error_vs_rain_rate.png")


if __name__ == "__main__":
    pars_folder = Path(
        "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
    )
    stereo_folder = Path(
        "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
    )
    parsivel_events = [
        pars_read(file_path) for file_path in sorted(pars_folder.iterdir())
    ]
    stereo_events = [
        pars_read(file_path) for file_path in sorted(stereo_folder.iterdir())
    ]
    print("READ THE DATA FOR BOTH DEVICES")
    main(parsivel_events, stereo_events)
    print("DONE.")
