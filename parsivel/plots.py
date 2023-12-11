from matplotlib.axes import Axes
from numpy import fromiter, linspace
from parsivel.parsivel_dataclass import ParsivelTimeSeries

from plots.styles import BASEPARSIVELSTYLE

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
    y = series.rain_rate
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
