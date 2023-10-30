from typing import List, Tuple
from matplotlib.axes import Axes
from parsivel.parsivel_dataclass import ParsivelTimeSeries
from aux_funcs.bin_data import CLASSES_DIAMETER, bin_diameter, bin_velocity
import numpy as np
import seaborn as sns
from aux_funcs.general import V_D_Lhermitte_1988
from collections import namedtuple
import matplotlib.colors as mcolors

LineStyle = namedtuple("LineStyle", ["ls", "c", "s"])

"""
Style of lines for bowth devices
"""
LINESTYLES = {
    "3D Stereo": LineStyle("dashdot", "magenta", 2.0),
    "Parsivel": LineStyle("solid", "green", 1.5),
}

###############################################################################
#################### AUXILIARY FUNCTIONS FOR PLOTTING #########################
###############################################################################
"""
Provide the ticks and tick_labels for a time series plot, to have labels spaced
every 6 hours.
"""


def get_ticks_n_labels(series: ParsivelTimeSeries):
    duration_s = series.duration[1] - series.duration[0]
    time_step_s = series.resolution_seconds
    time_step_6h = 3600 // time_step_s * 6
    ticks = [i for i in range(0, duration_s, time_step_6h)]
    labels = [f"{i*time_step_s//3600}h" for i in range(0, duration_s, time_step_6h)]

    return (ticks, labels)


"""
For the D x V graph, provide the hermitter tendency line 
"""


def get_hermitter_line() -> Tuple[np.ndarray, np.ndarray]:
    # Get an array of the diameters
    diameters = np.linspace(CLASSES_DIAMETER[0][0], CLASSES_DIAMETER[-1][0], 100)
    binned_velocities = np.array(
        [bin_velocity(n) for n in map(V_D_Lhermitte_1988, diameters)]
    )
    binned_diameters = np.array([bin_diameter(n) for n in diameters])

    return (binned_diameters, binned_velocities)


###############################################################################
########################## FUNCTIONS FOR PLOTTING #############################
###############################################################################
# TODO: Create a title with the field and the beggining and end

"""
The plot for the rain rate
"""


def plot_rain_rate(ax: Axes, list_series: List[ParsivelTimeSeries]):
    assert len(list_series) > 0, "Need at least 1 series to plot"
    first = list_series[0]

    # Custumise the axis
    ax.set_ylabel("R (mm.h^-1)")
    ax.set_xlabel("Time(h)")
    ticks, labels = get_ticks_n_labels(first)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    # Plot the graph
    for series in list_series:
        style = LINESTYLES[series.device]
        ax.plot(
            series.calculated_rate,
            linestyle=style.ls,
            c=style.c,
            linewidth=style.s,
            label=series.device,
        )
    ax.legend()


"""
Plot the Diameter x Velocity graph for the parsivel
"""


def plot_vxd(axs: List[Axes], series: List[ParsivelTimeSeries]):
    # Checks if there was a series sent
    assert len(series) > 0, "Need at least 1 series to plot"

    # Checks if the number of series is the same of axis
    assert len(axs) == len(series), "The number of axis has to be the same as series"

    right = max((np.max(serie.matrix_for_event) for serie in series))
    norm = mcolors.Normalize(0, right)

    lnx, lny = get_hermitter_line()
    for i, (ax, serie) in enumerate(zip(axs, series)):
        total_matrix = serie.matrix_for_event
        assert isinstance(total_matrix, np.ndarray)
        # Plot the tendency line
        ax.plot(lnx, lny, linewidth=1.6, linestyle="dotted", c="orangered")
        ax.set_title(serie.device)
        ticks, labels = [i + 0.5 for i in range(32)], [f"{i}" for i in range(1, 33)]

        # Create the heatmap for the classes
        cbar = False if i != len(series) - 1 else True
        sns.heatmap(
            total_matrix.T,
            norm=norm,
            ax=ax,
            cbar=cbar,
            cmap="bone_r",
            square=True,
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()


"""
The plot for the Cumulative Rain Depth
"""


def plot_cumulative_depth(ax: Axes, list_series: List[ParsivelTimeSeries]):
    # Assert the there is at leat one series
    assert len(list_series) > 0, "Need at least 1 series to plot"

    first = list_series[0]
    # Custumise the main axis
    ax.set_ylabel("Depth $(mm)$")
    ax.set_xlabel("Time(h)")
    ticks, labels = get_ticks_n_labels(first)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    # Plot all the series in the Axis
    for series in list_series:
        style = LINESTYLES[series.device]
        ax.plot(
            series.calculated_rain_depth,
            linestyle=style.ls,
            c=style.c,
            linewidth=style.s,
            label=series.device,
        )
    ax.legend()


"""
The plot for the Drops Size Distribution (dsd)
"""


def plot_dsd(ax: Axes, list_series: List[ParsivelTimeSeries]):
    # Assert the there is at leat one series
    assert len(list_series) > 0, "Need at least 1 series to plot"

    # Custumise the main axis
    ax.set_ylabel("$N(D)D^{3}$")
    ax.set_xlabel("$D$ $(mm)$")

    # Plot all the series in the Axis
    for series in list_series:
        # First get the overall matrix for that period
        overall_matrix = series.matrix_for_event

        # Get the diameters for each class
        mean_diameters = np.array([d[0] for d in CLASSES_DIAMETER])
        count_per_class = np.sum(overall_matrix, axis=1)
        nd3 = np.array(
            [md**3 * n for (md, n) in zip(mean_diameters, count_per_class)]
        )

        style = LINESTYLES[series.device]
        ax.plot(
            mean_diameters,
            nd3,
            linestyle=style.ls,
            c=style.c,
            linewidth=style.s,
            label=series.device,
        )
    ax.legend()


"""
Plot for the number of drops on time graph
"""


def plot_ndrops(ax: Axes, series: ParsivelTimeSeries):
    duration_s = series.duration[1] - series.duration[0]
    time_step_s = series.resolution_seconds
    # Plot the graphics
    # X axis
    ax.set_title(series.device)
    ax.set_xlabel("Time(h)")
    time_step_6h = 3600 // time_step_s * 6
    ax.set_xticks([i for i in range(0, duration_s, time_step_6h)])
    ax.set_xticklabels(
        [f"{i*time_step_s//3600}h" for i in range(0, duration_s, time_step_6h)]
    )

    style = LINESTYLES[series.device]
    ax.plot(
        [np.sum(matrix) for matrix in series.matrices],
        linestyle=style.ls,
        c=style.c,
        linewidth=style.s,
    )
