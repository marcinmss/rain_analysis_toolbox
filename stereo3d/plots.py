from matplotlib.axes import Axes
from numpy import fromiter, linspace

from plots.styles import BASEPARSIVELSTYLE
from stereo3d.stereo3d_dataclass import Stereo3DSeries

"""
Functions to plot important fields for the parsivel
"""


def plot_rain_rate(
    series: Stereo3DSeries,
    ax: Axes,
    style: dict = BASEPARSIVELSTYLE,
    resolution_seconds: int = 30,
):
    # Set the axis apperence
    ax.set_title("Rain Rate")
    ax.set_ylabel(r"$mm.h^{-1}$")
    ax.set_xlabel(r"$min$")

    # Plot the data
    y = series.rain_rate(resolution_seconds)
    x = fromiter(range(y.size), dtype=float)
    ax.plot(x, y, **style)

    # Set the ticks for the x axis
    n_ticks = 7
    ticks = linspace(0, y.size, n_ticks).tolist()
    ax.set_xticks(ticks)
    ticks_labels = [f"{int(t* resolution_seconds // 60)}" for t in ticks]
    ax.set_xticklabels(ticks_labels)
