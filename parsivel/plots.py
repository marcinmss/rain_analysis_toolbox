from typing import Tuple
from matplotlib.axes import Axes
from parsivel.parsivel_dataclass import ParsivelTimeSeries
from aux_funcs.bin_data import CLASSES_DIAMETER, bin_velocity
import numpy as np
import seaborn as sns


"""
The plot for the rain rate
"""


def plot_rain_rate(ax: Axes, series: ParsivelTimeSeries, title: str = ""):
    duration_s = series.duration[1] - series.duration[0]
    time_step_s = series.resolution_seconds
    # Plot the graphics
    ax.set_title(title)
    # X axis
    ax.set_xlabel("Time(h)")
    time_step_6h = 3600 // time_step_s * 6
    ax.set_xticks([i for i in range(0, duration_s, time_step_6h)])
    ax.set_xticklabels(
        [f"{i*time_step_s//3600}h" for i in range(0, duration_s, time_step_6h)]
    )

    # Y axis
    ax.set_ylabel("R (mm.h^-1)")

    ax.plot(series.calculated_rate2)


"""
The plot for the rain rate
"""


def plot_cumulative_depth(ax: Axes, series: ParsivelTimeSeries, title: str = ""):
    duration_s = series.duration[1] - series.duration[0]
    time_step_s = series.resolution_seconds
    # Plot the graphics
    ax.set_title(title)
    # X axis
    ax.set_xlabel("Time(h)")
    time_step_6h = 3600 // time_step_s * 6
    ax.set_xticks([i for i in range(0, duration_s, time_step_6h)])
    ax.set_xticklabels(
        [f"{i*time_step_s//3600}h" for i in range(0, duration_s, time_step_6h)]
    )

    # Y axis
    ax.set_ylabel("Depth $(mm)$")

    ax.plot(series.calculated_rain_depth2)


"""
Plot for the number of drops on time graph
"""


def plot_ndrops(ax: Axes, series: ParsivelTimeSeries):
    duration_s = series.duration[1] - series.duration[0]
    time_step_s = series.resolution_seconds
    # Plot the graphics
    # X axis
    ax.set_xlabel("Time(h)")
    time_step_6h = 3600 // time_step_s * 6
    ax.set_xticks([i for i in range(0, duration_s, time_step_6h)])
    ax.set_xticklabels(
        [f"{i*time_step_s//3600}h" for i in range(0, duration_s, time_step_6h)]
    )
    # Color the plot
    ax.set_title("Number of Drops")

    ax.plot([np.sum(matrix) for matrix in series.get_sdd_matrix])


"""
The plot for the dsd graph
"""


def plot_nd3xd(ax: Axes, series: ParsivelTimeSeries, title: str = ""):
    ax.set_title(title)
    ax.set_ylabel("$N(D)D^{3}$")
    ax.set_xlabel("$D$ $(mm)$")

    # First get the overall matrix for that period
    overall_matrix = series.get_overall_matrix

    # Get the diameters for each class
    mean_diameters = np.array([d[0] for d in CLASSES_DIAMETER])
    count_per_class = np.sum(overall_matrix, axis=1)
    nd3 = np.array([md**3 * n for (md, n) in zip(mean_diameters, count_per_class)])

    ax.plot(mean_diameters, nd3)


"""
Plot velocity vs diameter
"""


def V_D_Lhermitte_1988(d_mm: float) -> float:
    d_mm *= 1e-3
    return 9.25 * (1 - np.exp(-1 * (68000 * (d_mm**2) + 488 * d_mm)))


def get_hermitter_line() -> Tuple[np.ndarray, np.ndarray]:
    # Get an array of the diameters
    diameters = np.linspace(CLASSES_DIAMETER[0][0], CLASSES_DIAMETER[-1][0], 1000)
    binned_velocities = np.array(
        [bin_velocity(n) for n in map(V_D_Lhermitte_1988, diameters)]
    )
    binned_diameters = np.array([bin_velocity(n) for n in diameters])

    return (binned_diameters, binned_velocities)


def plot_vxd(ax: Axes, series: ParsivelTimeSeries, title: str = ""):
    total_matrix = series.get_overall_matrix
    assert isinstance(total_matrix, np.ndarray)
    # Plot the tendency line
    lnx, lny = get_hermitter_line()
    ax.plot(lnx, lny)
    ax.set_title(title)

    # Create the heatmap for the classes
    sns.heatmap(total_matrix.T, cmap="hot_r", ax=ax)
    ax.invert_yaxis()
