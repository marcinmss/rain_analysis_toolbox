from typing import List
from parsivel import parsivel_read_from_pickle as pars_read
from matplotlib import pyplot as plt
from parsivel import ParsivelTimeSeries
from pathlib import Path
from numpy import floor, linspace, zeros

output_folder = Path(__file__).parent / "output"


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
    y_limts = linspace(LEFT, RIGHT, RIGHT + 1)
    x = (y_limts[:-1] + y_limts[1:]) / 2

    y_stereo, y_parsivel = zeros(x.shape), zeros(x.shape)

    area_stereo = stereo_converted_events[0].area_of_study
    area_parsivel = parsivel_events[0].area_of_study
    for stereo_tstep, parsivel_tstep in zip(stereo_time_steps, parsivel_time_steps):
        stereo_rain_rate = stereo_tstep.volume_mm3 / area_stereo * 120
        y_stereo[int((stereo_rain_rate - LEFT) // 1)] += stereo_rain_rate / 120
        parsivel_rain_rate = parsivel_tstep.volume_mm3 / area_parsivel * 120
        y_parsivel[int((parsivel_rain_rate - LEFT) // 1)] += parsivel_rain_rate / 120

    y_error = (y_parsivel - y_stereo) / y_parsivel

    # Plot both graphs
    figure = plt.figure()
    figure.set_size_inches((2 * 3 + 2, 1 * 3 + 1))
    figure.set_layout_engine("constrained")
    figure.suptitle("Analisys based on rain rate", fontsize=16)

    # Plot the first graph
    ax = figure.add_subplot(1, 2, 1)
    ax.plot(x, y_stereo, c="dodgerblue")
    print(f"max_y_stereo = {max(y_stereo)}")
    ax.plot(x, y_parsivel, c="orangered")
    ax.set_xlabel("rain rate $(mm.h^{-1})$", fontdict={"fontsize": 13})
    ax.set_ylabel("cummulative depth $(mm)$", fontdict={"fontsize": 13})
    ax.set_xbound(-0.5, 80)

    # Plot the first graph
    ax = figure.add_subplot(1, 2, 2)
    ax.plot(x, y_error, c="dodgerblue")
    ax.set_xlabel("rain rate $(mm.h^{-1})$", fontdict={"fontsize": 13})
    ax.set_ylabel("error", fontdict={"fontsize": 13})

    figure.savefig(output_folder / "depth_vs_rain_rate_.png")


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