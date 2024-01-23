from pathlib import Path
from typing import List

from numpy import fromiter, linspace, mean, ndarray, zeros
from matplotlib.pyplot import figure

from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle
from parsivel.indicators import mean_diameter_matrix, matrix_to_volume

OUTPUTFOLDER = Path(__file__).parent / "output"

parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
)

stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
)


def moble_average(field: ndarray, window_size: int, order: int = 1):
    if order < 1:
        return field
    else:
        new_array = fromiter(
            (mean(field[i : i + window_size]) for i in range(field.size)), dtype=float
        )
        return moble_average(new_array, window_size, order - 1)


def get_idx_func(mindiam: float, bin_size: float):
    def idx_func(diam: float) -> int:
        return int((diam - mindiam) // bin_size)

    return idx_func


def calculate_total_depth_vs_dm(events: List[ParsivelTimeSeries]):
    # Filter the converted events for the drops outside parsivel resolution
    area = events[0].area_of_study
    matrices = (tstep.matrix for event in events for tstep in event)

    MINDIAM, MAXDIAM, NBINS = -5, 10, 400
    BINSIZE = (MAXDIAM - MINDIAM) / NBINS
    bin_lims = linspace(MINDIAM, MAXDIAM, NBINS + 1)
    x = (bin_lims[1:] + bin_lims[:-1]) / 2
    y = zeros(x.shape, dtype=float)
    idx_func = get_idx_func(MINDIAM, BINSIZE)
    for matrix in matrices:
        y[idx_func(mean_diameter_matrix(matrix))] += matrix_to_volume(matrix) / area
    return (x, y)


if __name__ == "__main__":
    # Read the data for all 3 types
    print("Reading Parsivel Data")
    parsivel_events = [
        parsivel_read_from_pickle(file)
        for file in sorted(parsivel_events_folder.iterdir())
    ]
    print("Done.")
    print("Calculating for Parsivel")
    x, y_parsivel = calculate_total_depth_vs_dm(parsivel_events)
    del parsivel_events
    print("Done.")

    print("Reading 3D Stereo Data")
    stereo_events = [
        parsivel_read_from_pickle(file).filter_by_parsivel_resolution()
        for file in sorted(stereo_events_folder.iterdir())
    ]
    print("Done.")
    print("Calculating for stereo")
    x, y_stereo = calculate_total_depth_vs_dm(stereo_events)
    del stereo_events
    print("Done.")

    fig = figure()
    fig.set_size_inches(8.5, 6.5)

    # dsd per class
    ws = 10
    ord = 5
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        moble_average(x, ws, ord), moble_average(y_stereo, ws, ord), color="dodgerblue"
    )
    ax.plot(
        moble_average(x, ws, ord), moble_average(y_parsivel, ws, ord), color="orangered"
    )

    ax.set_title("Drop size distribution")
    ax.set_ylabel("Cumulative Depth")
    ax.set_xlabel("$D_m$")
    ax.set_xbound(0, 5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.75, 0.25), fontsize=16)
    fig.savefig(OUTPUTFOLDER / "depth_vs_dm.png")
