from typing import List
from parsivel import parsivel_read_from_pickle as pars_read
from matplotlib import pyplot as plt
from parsivel import ParsivelTimeSeries
import numpy as np
from pathlib import Path

output_folder = Path(__file__).parent / "output"


def plot_identety(ax):
    maximum_value = max(max(ax.get_xbound()), max(ax.get_ybound()))
    ax.plot((0, maximum_value), (0, maximum_value))


def overall_analysis(
    parsivel_events: List[ParsivelTimeSeries],
    stereo_converted_events: List[ParsivelTimeSeries],
):
    dotstyle = {"s": 14.0, "c": "orangered", "marker": "."}

    # Plot the rain rate
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_title("Average Rain Rate $(mm.h^{-1})$", fontdict={"fontsize": 14})
    ax.set_ylabel("Parsivel", fontdict={"fontsize": 13})
    ax.set_xlabel("Stereo 3D", fontdict={"fontsize": 13})
    pars_values = [np.mean(event.rain_rate()) for event in parsivel_events]
    stereo_values = [np.mean(event.rain_rate()) for event in stereo_converted_events]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    figure.savefig(output_folder / "summary_rain_rate.png")

    # Plot the total rain depth
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_title("Depth $(mm)$", fontdict={"fontsize": 14})
    ax.set_ylabel("Parsivel", fontdict={"fontsize": 13})
    ax.set_xlabel("Stereo 3D", fontdict={"fontsize": 13})
    pars_values = [event.total_depth_for_event() for event in parsivel_events]
    stereo_values = [event.total_depth_for_event() for event in stereo_converted_events]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    figure.savefig(output_folder / "summary_depth.png")

    # Plot the Kinecti energy per Area
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_title("Kinetic energy per Area $(j.m^{-2})$", fontdict={"fontsize": 14})
    ax.set_ylabel("Parsivel", fontdict={"fontsize": 13})
    ax.set_xlabel("Stereo 3D", fontdict={"fontsize": 13})
    pars_values = [event.kinetic_energy_flow_for_event() for event in parsivel_events]
    stereo_values = [
        event.kinetic_energy_flow_for_event() for event in stereo_converted_events
    ]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    figure.savefig(output_folder / "summary_kinetic.png")

    # Plot the mean diameter
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_title("Number of drops per area $(m^{-2})$", fontdict={"fontsize": 14})
    ax.set_ylabel("Parsivel", fontdict={"fontsize": 13})
    ax.set_xlabel("Stereo 3D", fontdict={"fontsize": 13})
    pars_values = [
        event.get_number_drops() / event.area_of_study for event in parsivel_events
    ]
    stereo_values = [
        event.get_number_drops() / event.area_of_study
        for event in stereo_converted_events
    ]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    figure.savefig(output_folder / "summary_ndrops.png")

    # Plot the mean diameter
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_title("Mean Diameter $(mm)$", fontdict={"fontsize": 14})
    ax.set_ylabel("Parsivel", fontdict={"fontsize": 13})
    ax.set_xlabel("Stereo 3D", fontdict={"fontsize": 13})
    pars_values = [event.mean_diameter_for_event() for event in parsivel_events]
    stereo_values = [
        event.mean_diameter_for_event() for event in stereo_converted_events
    ]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    figure.savefig(output_folder / "summary_mean_diameter.png")

    # Plot the mean velocity
    figure = plt.figure()
    figure.set_layout_engine("constrained")
    figure.set_size_inches((1 * 3 + 2, 1 * 3 + 1))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_title("Mean Velocity $(m.s^{-1})$", fontdict={"fontsize": 14})
    ax.set_ylabel("Parsivel", fontdict={"fontsize": 13})
    ax.set_xlabel("Stereo 3D", fontdict={"fontsize": 13})
    pars_values = [event.mean_velocity_for_event() for event in parsivel_events]
    stereo_values = [
        event.mean_velocity_for_event() for event in stereo_converted_events
    ]
    ax.scatter(stereo_values, pars_values, **dotstyle)
    plot_identety(ax)
    figure.savefig(output_folder / "summary_mean_velocity.png")


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
    overall_analysis(parsivel_events, stereo_events)
    print("DONE.")
