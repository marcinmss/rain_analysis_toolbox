from matplotlib.pyplot import figure
from individual_analysis.analysis_variables import FIGURESPECS, LEGENDFIGURESPECS
import numpy as np
from pathlib import Path
from parsivel.read_write import parsivel_read_from_pickle

from stereo import stereo_read_from_pickle

STEREOEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/stereo_full_event.obj"
)
PARSIVELEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/parsivel_full_event.obj"
)
OUTPUTFOLDER = Path(__file__).parent / "output/"

AXESTITLEFONTSIZE = 18
AXESLABELFONTSIZE = 16

avg_parsivel_lnstyle = {
    "color": "orangered",
    "label": "Average Parsivel",
    "linestyles": "--",
    "alpha": 0.7,
}
avg_stereo_lnstyle = {
    "color": "dodgerblue",
    "label": "Average 3D Stereo",
    "linestyles": "--",
    "alpha": 0.7,
}


def plot_avg_for_both_devices(ax, stereo_avg, parsivel_avg):
    ax.hlines(
        y=stereo_avg,
        xmin=1,
        xmax=9,
        **avg_stereo_lnstyle,
    )
    ax.hlines(
        y=parsivel_avg,
        xmin=1,
        xmax=9,
        **avg_parsivel_lnstyle,
    )


if __name__ == "__main__":
    # Read the data for both devices
    print("Reading the full event stereo.")
    stereo_event = stereo_read_from_pickle(STEREOEVENTSFOLDER)
    print("Splitting the full event in sections")
    sections = stereo_event.split_by_distance_to_sensor()

    print("Reading the full event parsivel.")
    parsivel_event = parsivel_read_from_pickle(PARSIVELEVENTSFOLDER)

    x_values = np.array([i for i in range(1, 9)])
    print("Split the full event in sections")

    # Create the figure
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()

    y_sections = np.array([len(event) / event.area_of_study for event in sections])
    stereo_avg = len(stereo_event) / stereo_event.area_of_study
    parsivel_avg = len(stereo_event) / stereo_event.area_of_study

    plot_avg_for_both_devices(ax, stereo_avg, parsivel_avg)
    ax.scatter(x=x_values, y=y_sections, color="dodgerblue", label="Value for Section")

    # Custumize axis and save figure
    ax.set_title("Number of drops per area", fontsize=AXESTITLEFONTSIZE)
    ax.set_ylabel("$mm^{-2}$", fontsize=AXESLABELFONTSIZE)
    ax.set_xlabel("Section number", fontsize=AXESLABELFONTSIZE)
    ax.set_ybound(lower=0)
    fig.savefig(OUTPUTFOLDER / "sda_summary_ndrops.png")

    # Plot the Total Depth
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()

    y_sections = np.array([event.total_depth_for_event() for event in sections])
    stereo_avg = stereo_event.total_depth_for_event()
    parsivel_avg = parsivel_event.total_depth_for_event()

    plot_avg_for_both_devices(ax, stereo_avg, parsivel_avg)
    ax.scatter(x=x_values, y=y_sections, color="dodgerblue", label="Value for Section")

    ax.set_title("Total depht", fontsize=AXESTITLEFONTSIZE)
    ax.set_ylabel("$mm$", fontsize=AXESLABELFONTSIZE)
    ax.set_xlabel("Section number", fontsize=AXESLABELFONTSIZE)
    ax.set_ybound(lower=0)
    fig.savefig(OUTPUTFOLDER / "sda_summary_total_depth.png")

    # Plot the Kinetic per Area
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()

    y_sections = np.array([event.kinetic_energy_flow() for event in sections])
    stereo_avg = stereo_event.kinetic_energy_flow()
    parsivel_avg = parsivel_event.kinetic_energy_flow_for_event()

    plot_avg_for_both_devices(ax, stereo_avg, parsivel_avg)
    ax.scatter(x=x_values, y=y_sections, color="dodgerblue", label="Value for Section")

    ax.set_title("Kinetic energy per area", fontsize=AXESTITLEFONTSIZE)
    ax.set_ylabel("$J.m^{-2}$", fontsize=AXESLABELFONTSIZE)
    ax.set_xlabel("Section number", fontsize=AXESLABELFONTSIZE)
    ax.set_ybound(lower=0)
    fig.savefig(OUTPUTFOLDER / "sda_summary_kinetic.png")

    # Plot the Mean Diameter
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()

    y_sections = np.array([event.mean_diameter_for_event() for event in sections])
    stereo_avg = stereo_event.mean_diameter_for_event()
    parsivel_avg = parsivel_event.mean_diameter_for_event()

    plot_avg_for_both_devices(ax, stereo_avg, parsivel_avg)
    ax.scatter(x=x_values, y=y_sections, color="dodgerblue", label="Value for Section")

    ax.set_title("Mean diameter", fontsize=AXESTITLEFONTSIZE)
    ax.set_ylabel("$mm$", fontsize=AXESLABELFONTSIZE)
    ax.set_xlabel("Section number", fontsize=AXESLABELFONTSIZE)
    ax.set_ybound(lower=0)
    fig.savefig(OUTPUTFOLDER / "sda_summary_mean_diameter.png")

    # Plot the Mean Velocity
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()

    y_sections = np.array([event.mean_velocity_for_event() for event in sections])
    stereo_avg = stereo_event.mean_velocity_for_event()
    parsivel_avg = parsivel_event.mean_velocity_for_event()

    plot_avg_for_both_devices(ax, stereo_avg, parsivel_avg)
    ax.scatter(x=x_values, y=y_sections, color="dodgerblue", label="Value for Section")

    ax.set_title("Mean velocity", fontsize=AXESTITLEFONTSIZE)
    ax.set_ylabel("$mm$", fontsize=AXESLABELFONTSIZE)
    ax.set_xlabel("Section number", fontsize=AXESLABELFONTSIZE)
    ax.set_ybound(lower=0)
    fig.savefig(OUTPUTFOLDER / "sda_summary_mean_velocity.png")

    # Plot the labels
    handles, labels = ax.get_legend_handles_labels()
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    ax.set_axis_off()
    ax.legend(handles, labels, **LEGENDFIGURESPECS)
    fig.savefig(OUTPUTFOLDER / "sda_summary_labels.png")
