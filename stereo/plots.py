from matplotlib.axes import Axes
from numpy import fromiter, linspace, zeros

from plots.styles import BASEPARSIVELSTYLE, BASESTEREOSTYLE
from stereo.dataclass import Stereo3DSeries

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


"""
Plot number of drops per area
"""


def plot_dsd(
    series: Stereo3DSeries,
    ax: Axes,
    style: dict = BASESTEREOSTYLE,
):
    # Calculate the data
    nbins = 100
    bins_limits = linspace(0, 27.5, nbins + 1)
    bin_size = bins_limits[1]
    x = (bins_limits[:-1] + bins_limits[1:]) / 2
    y = zeros(x.shape)
    duration_seconds = series.duration[1] - series.duration[0]
    constant_factor = 1 / (series.area_of_study * bin_size * duration_seconds)
    for drop in series:
        y[int(drop.diameter // bin_size)] += 1 / drop.velocity * constant_factor

    # Plot the data
    # Set the axis apperence
    ax.set_title("put title")
    ax.set_ylabel("ylabel")
    ax.set_xlabel("xlabel")

    ## plot
    ax.plot(x, y, **style)
