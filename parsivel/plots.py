from matplotlib.axes import Axes
from numpy import fromiter, linspace, ndarray, zeros
from parsivel.dataclass import ParsivelTimeSeries, BASEPARSIVELSTYLE
from parsivel.matrix_classes import (
    CLASSES_DIAMETER_MIDDLE,
    CLASSES_DIAMETER,
    CLASSES_VELOCITY_MIDDLE,
)

"""
Functions to plot important fields for the parsivel
"""


def plot_rain_rate(
    series: ParsivelTimeSeries, ax: Axes, style: dict = BASEPARSIVELSTYLE
):
    # Set the axis apperence
    ax.set_title("Rain Rate")
    ax.set_ylabel(r"$mm.h^{-1}$")
    ax.set_xlabel(r"$min$")

    # Plot the data
    y = series.rain_rate()
    x = fromiter((tstep.timestamp for tstep in series), dtype=float)
    ax.plot(x, y, **style)

    # Set the ticks for the x axis
    n_ticks = 7
    ticks = linspace(series.duration[0], series.duration[1], n_ticks).tolist()
    ax.set_xticks(ticks)
    ticks_labels = [
        f"{int((t - series.duration[0]) * series.resolution_seconds // 60)}"
        for t in ticks
    ]
    ax.set_xticklabels(ticks_labels)


"""
Plot n(d) of drops per area
"""


def plot_dsd(
    series: ParsivelTimeSeries,
    ax: Axes,
    style: dict = BASEPARSIVELSTYLE,
):
    # Get relevant variables
    matrix = series.matrix_for_event
    assert isinstance(matrix, ndarray)
    dt = series.duration[1] - series.duration[0]
    area_m2 = series.area_of_study * 1e-6

    # Calculate the data
    y = zeros(32, dtype=float)
    x = zeros(32, dtype=float)
    for idx_diam in range(32):
        mult_factor = 1 / (area_m2 * dt * CLASSES_DIAMETER[idx_diam][1])
        x[idx_diam] = CLASSES_DIAMETER_MIDDLE[idx_diam]
        y[idx_diam] = (
            sum(
                matrix[idx_vel, idx_diam] / CLASSES_VELOCITY_MIDDLE[idx_vel]
                for idx_vel in range(32)
            )
            * mult_factor
        )

    # Plot the data
    # Set the axis apperence
    ax.set_title("Drop Size distribution")
    ax.set_ylabel("$N(d)$")
    ax.set_xlabel("diameter $(mm)$")

    ## plot
    ax.plot(x, y, **style)
    ax.legend()
