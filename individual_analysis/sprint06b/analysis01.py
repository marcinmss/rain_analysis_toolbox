from typing import List
from stereo3d import stereo_read_from_pickle as std_read
from parsivel import pars_read_from_pickle as pars_read
from matplotlib import pyplot as plt
from parsivel import ParsivelTimeSeries
from stereo3d import Stereo3DSeries
import numpy as np
from pathlib import Path

output_folder = Path("/home/marcio/stage_project/individual_analysis/sprint06b/output/")


def plot_identety(ax):
    maximum_value = max(max(ax.get_xbound()), max(ax.get_ybound()))
    ax.plot((0, maximum_value), (0, maximum_value))


def overall_analysis(
    parsivel_events: List[ParsivelTimeSeries], stereo_events: List[Stereo3DSeries]
):
    dotstyle = {"s": 18.0, "c": "orangered", "marker": "."}
    ncols = 3
    nrows = 2
    figure = plt.figure()
    figure.set_size_inches((ncols * 3 + 2, nrows * 3 + 1))
    figure.set_layout_engine("constrained")
    figure.suptitle("Analisys for multiple events", fontsize=18)
    plot_idx = 1

    # Plot the rain rate
    ax = figure.add_subplot(nrows, ncols, plot_idx)
    ax.set_title("Average Rain Rate $(mm.h^{-1})$")
    ax.set_ylabel("Parsivel")
    ax.set_xlabel("Stereo 3D")
    pars_values = [np.mean(event.rain_rate) for event in parsivel_events]
    stereo_values = [np.mean(event.rain_rate()) for event in stereo_events]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    plot_idx += 1

    # Plot the total rain depth
    ax = figure.add_subplot(nrows, ncols, plot_idx)
    ax.set_title("Total depth $(mm)$")
    ax.set_ylabel("Parsivel")
    ax.set_xlabel("Stereo 3D")
    pars_values = [event.total_depth_for_event for event in parsivel_events]
    stereo_values = [event.total_depth_for_event for event in stereo_events]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    plot_idx += 1

    # Plot the Kinecti energy per Area
    ax = figure.add_subplot(nrows, ncols, plot_idx)
    ax.set_title("Kinetic energy per Area $(j.m^{-2})$")
    ax.set_ylabel("Parsivel")
    ax.set_xlabel("Stereo 3D")
    pars_values = [event.kinetic_energy_flow_for_event for event in parsivel_events]
    stereo_values = [event.kinetic_energy_flow for event in stereo_events]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    plot_idx += 1

    # Plot the mean diameter
    ax = figure.add_subplot(nrows, ncols, plot_idx)
    ax.set_title("Number of drops per area $(m^{-2})$")
    ax.set_ylabel("Parsivel")
    ax.set_xlabel("Stereo 3D")
    pars_values = [event.npa_for_event for event in parsivel_events]
    stereo_values = [event.npa_event for event in stereo_events]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    plot_idx += 1

    # Plot the mean diameter
    ax = figure.add_subplot(nrows, ncols, plot_idx)
    ax.set_title("Mean Diameter $(mm)$")
    ax.set_ylabel("Parsivel")
    ax.set_xlabel("Stereo 3D")
    pars_values = [event.mean_diameter_for_event for event in parsivel_events]
    stereo_values = [event.mean_diameter_for_event for event in stereo_events]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    plot_idx += 1

    # Plot the mean velocity
    ax = figure.add_subplot(nrows, ncols, plot_idx)
    ax.set_title("Mean Velocity $(m.s^{-1})$")
    ax.set_ylabel("Parsivel")
    ax.set_xlabel("Stereo 3D")
    pars_values = [event.mean_velocity_for_event for event in parsivel_events]
    stereo_values = [event.mean_velocity_for_event for event in stereo_events]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    plot_idx += 1

    figure.savefig(output_folder / "overall_comparison.png")


if __name__ == "__main__":
    pars_folder = Path(
        "/home/marcio/stage_project/data/saved_events/sprint05/parsivel/"
    )
    stereo_folder = Path(
        "/home/marcio/stage_project/data/saved_events/sprint05/stereo/"
    )
    parsivel_events = [pars_read(file_path) for file_path in pars_folder.iterdir()]
    stereo_events = [std_read(file_path) for file_path in stereo_folder.iterdir()]
    print("READ THE DATA FOR BOTH DEVICES")
    overall_analysis(parsivel_events, stereo_events)
    print("DONE.")