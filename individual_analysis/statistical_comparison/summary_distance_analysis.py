from typing import List
from parsivel import parsivel_read_from_pickle as pars_read
from matplotlib import pyplot as plt
from parsivel import ParsivelTimeSeries
import numpy as np
from pathlib import Path
from matplotlib import colormaps
from parsivel.read_write import parsivel_read_from_pickle

from stereo import stereo_read_from_pickle
from stereo.dataclass import Stereo3DSeries

stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/stereo_full_event.obj"
)
parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/parsivel_full_event.obj"
)
output_folder = Path(__file__).parent / "output/"


def overall_analysis(stereo_event: Stereo3DSeries, parsivel_event: ParsivelTimeSeries):
    # Read the data for both devices
    sections = stereo_event.split_by_distance_to_sensor()
    x_values = np.array([i for i in range(1, 9)])
    parsivel_style = {
        "color": "orangered",
        "label": "Average Parsivel",
        "linestyles": "--",
    }
    stereo_style = {
        "color": "dodgerblue",
        "label": "Average 3D Stereo",
        "linestyles": "--",
    }
    print("Split the full event in sections")

    # Plot the Number of Drops per Area
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.hlines(
        y=len(stereo_event) / stereo_event.area_of_study, xmin=1, xmax=9, **stereo_style
    )
    ax.hlines(
        y=parsivel_event.ndrops() / parsivel_event.area_of_study,
        xmin=1,
        xmax=9,
        **parsivel_style
    )
    y_values = np.array([len(event) / event.area_of_study for event in sections])
    ax.scatter(x=x_values, y=y_values, color="dodgerblue", label="Value for Section")
    ax.set_title("Number of Drops per Area", fontdict={"fontsize": 16})
    ax.set_ylabel("$mm^{-2}$", fontdict={"fontsize": 14})
    ax.set_xlabel("section number", fontdict={"fontsize": 14})
    ax.set_ybound(lower=0)
    figure.savefig(output_folder / "distance_analysis_ndrops.png")

    # Plot the Total Depth
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.hlines(y=stereo_event.total_depth_for_event(), xmin=1, xmax=9, **stereo_style)
    ax.hlines(
        y=parsivel_event.total_depth_for_event(), xmin=1, xmax=9, **parsivel_style
    )
    y_values = np.array([event.total_depth_for_event() for event in sections])
    ax.scatter(x=x_values, y=y_values, color="dodgerblue", label="Value for Section")
    ax.set_title("Total Depht", fontdict={"fontsize": 16})
    ax.set_ylabel("$mm$", fontdict={"fontsize": 14})
    ax.set_xlabel("section number", fontdict={"fontsize": 14})
    ax.set_ybound(lower=0)
    figure.savefig(output_folder / "distance_analysis_total_depth.png")

    # Plot the Kinetic per Area
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.hlines(y=stereo_event.kinetic_energy_flow(), xmin=1, xmax=9, **stereo_style)
    ax.hlines(
        y=parsivel_event.kinetic_energy_flow_for_event(),
        xmin=1,
        xmax=9,
        **parsivel_style
    )
    y_values = np.array([event.kinetic_energy_flow() for event in sections])
    ax.scatter(x=x_values, y=y_values, color="dodgerblue", label="Value for Section")
    ax.set_title("Kinetic Energy by Area", fontdict={"fontsize": 16})
    ax.set_ylabel("$J.m^{-2}$", fontdict={"fontsize": 14})
    ax.set_xlabel("section number", fontdict={"fontsize": 14})
    ax.set_ybound(lower=0)
    figure.savefig(output_folder / "distance_analysis_kinetic.png")

    # Plot the Mean Diameter
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.hlines(y=stereo_event.mean_diameter_for_event(), xmin=1, xmax=9, **stereo_style)
    ax.hlines(
        y=parsivel_event.mean_diameter_for_event(), xmin=1, xmax=9, **parsivel_style
    )
    y_values = np.array([event.mean_diameter_for_event() for event in sections])
    ax.scatter(x=x_values, y=y_values, color="dodgerblue", label="Value for Section")
    ax.set_title("Mean Diameter", fontdict={"fontsize": 16})
    ax.set_ylabel("$mm$", fontdict={"fontsize": 14})
    ax.set_xlabel("section number", fontdict={"fontsize": 14})
    ax.set_ybound(lower=0)
    figure.savefig(output_folder / "distance_analysis_mean_diameter.png")

    # Plot the Mean Velocity
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.hlines(y=stereo_event.mean_velocity_for_event(), xmin=1, xmax=9, **stereo_style)
    ax.hlines(
        y=parsivel_event.mean_velocity_for_event(), xmin=1, xmax=9, **parsivel_style
    )
    y_values = np.array([event.mean_velocity_for_event() for event in sections])
    ax.scatter(x=x_values, y=y_values, color="dodgerblue", label="Value for Section")
    ax.set_title("Mean Velocity", fontdict={"fontsize": 16})
    ax.set_ylabel("$mm$", fontdict={"fontsize": 14})
    ax.set_xlabel("section number", fontdict={"fontsize": 14})
    ax.set_ybound(lower=0)
    figure.savefig(output_folder / "distance_analysis_mean_velocity.png")

    # Plot the labels
    handles, labels = ax.get_legend_handles_labels()
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.legend(handles, labels, loc="center", fontsize=14)
    figure.savefig(output_folder / "distance_analysis_labels.png")


if __name__ == "__main__":
    # Read the data for both devices
    stereo_event = stereo_read_from_pickle(stereo_events_folder)
    print("Read the full event stereo.")
    parsivel_event = parsivel_read_from_pickle(parsivel_events_folder)
    print("Read the full event parsivel.")
    overall_analysis(stereo_event, parsivel_event)
