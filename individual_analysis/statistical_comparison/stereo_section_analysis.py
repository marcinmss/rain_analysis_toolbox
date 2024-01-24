from parsivel import parsivel_read_from_pickle
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from aux_funcs.general import V_D_Lhermitte_1988
from sklearn.metrics import r2_score

from stereo.read_write import stereo_read_from_pickle

OUTPUTFOLDER = Path(__file__).parent / "output"


def plot_mean_values(figures_by_session, mean_pars, mean_stereo, ax: Axes):
    ytop = max((max(figures_by_session), mean_pars, mean_stereo))
    ax.scatter(sessions_numbers, figures_by_session, label="value_by_section")
    ax.set_xlabel("sections")
    ax.hlines(
        mean_stereo,
        1,
        8,
        colors="dodgerblue",
        linestyles="--",
        label="Average 3D Stereo",
    )
    ax.hlines(
        mean_pars, 1, 8, colors="orangered", linestyles="--", label="Average Parsivel"
    )
    ax.set_xbound(0.5, 8.5)
    ax.set_ybound(0, 1.05 * ytop)


parsivel_file_full_period = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/parsivel_full_event.obj"
)
stereo_file_full_period = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/stereo_full_event.obj"
)

if __name__ == "__main__":
    # Read the main parsivel series
    parsivel_event = parsivel_read_from_pickle(parsivel_file_full_period)
    stereo_event = stereo_read_from_pickle(stereo_file_full_period)
    stereo_sections = stereo_event.split_by_distance_to_sensor()

    # Plot the first graph
    figure = plt.figure()
    ncols, nrows = 3, 2
    figure.set_size_inches(13, nrows * 4)
    sessions_numbers = list(range(1, len(stereo_sections) + 1))

    counter = 1

    figure.set_layout_engine("constrained")

    # Number of drops per Area
    ax = figure.add_subplot(nrows, ncols, counter)
    indicator = [
        section.total_number_of_drops() / section.area_of_study
        for section in stereo_sections
    ]
    parsivel_mean = (
        parsivel_event.total_number_of_drops() / parsivel_event.area_of_study
    )
    stereo_mean = stereo_event.total_number_of_drops() / stereo_event.area_of_study
    plot_mean_values(indicator, parsivel_mean, stereo_mean, ax)
    ax.set_title("Number of drops per Area")
    ax.set_ylabel("$m^{-2}$")
    counter += 1

    # Total depth
    ax = figure.add_subplot(nrows, ncols, counter)
    indicator = [section.total_depth_for_event() for section in stereo_sections]
    parsivel_mean = parsivel_event.total_depth_for_event()
    stereo_mean = stereo_event.total_depth_for_event()
    plot_mean_values(indicator, parsivel_mean, stereo_mean, ax)
    ax.set_title("Total Depth")
    ax.set_ylabel("$mm$")
    counter += 1

    # Kinetic energy
    ax = figure.add_subplot(nrows, ncols, counter)
    indicator = [section.kinetic_energy_flow() for section in stereo_sections]
    parsivel_mean = parsivel_event.kinetic_energy_flow_for_event()
    stereo_mean = stereo_event.kinetic_energy_flow()
    plot_mean_values(indicator, parsivel_mean, stereo_mean, ax)
    ax.set_title("Kinetic Energy by area")
    ax.set_ylabel("$j.m^{-2}$")
    counter += 1

    # Mean Diameter
    ax = figure.add_subplot(nrows, ncols, counter)
    indicator = [section.mean_diameter_for_event() for section in stereo_sections]
    parsivel_mean = parsivel_event.mean_diameter_for_event()
    stereo_mean = stereo_event.mean_diameter_for_event()
    plot_mean_values(indicator, parsivel_mean, stereo_mean, ax)
    ax.set_title("Mean Diameter")
    ax.set_ylabel("$mm$")
    counter += 1

    # Mean Velocity
    ax = figure.add_subplot(nrows, ncols, counter)
    indicator = [section.mean_velocity_for_event() for section in stereo_sections]
    parsivel_mean = parsivel_event.mean_velocity_for_event()
    stereo_mean = stereo_event.mean_velocity_for_event()
    plot_mean_values(indicator, parsivel_mean, stereo_mean, ax)
    ax.set_title("Mean Velocity")
    ax.set_ylabel("$m.s^{-1}$")
    counter += 1

    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc=(0.75, 0.25), fontsize=16)
    figure.savefig(OUTPUTFOLDER / "summary_sections_analysis.png")

    # Plot the second graph
    figure = plt.figure()
    figure.set_size_inches((14, 6))
    figure.set_layout_engine("constrained")

    lnx = np.linspace(0, 4, 100)
    lny = V_D_Lhermitte_1988(lnx)

    for i, section in enumerate(stereo_sections):
        ax = figure.add_subplot(2, 4, i + 1)
        ax.scatter(
            section.diameters,
            section.velocity,
            marker=".",
            s=2.0,
        )

        r2 = r2_score(V_D_Lhermitte_1988(section.diameters), section.velocity)
        percentage_bellow = (
            sum(
                1
                for item in section
                if V_D_Lhermitte_1988(item.diameter) > item.velocity
            )
            / len(section)
            * 100
        )
        textstr = "\n".join(
            (r"$R^{2}=%.2f$" % (r2,), r"pbl$=%.1f$" % (percentage_bellow,))
        )

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_xlabel("diameter $(mm)$")
        ax.set_ylabel("velocity $(m.s^{-1})$")

        ax.set_title(f"Section {i+ 1}")
        ax.plot(lnx, lny, "r--")
        ax.set_ybound(0, 10)
        ax.set_xbound(0, 5)

    figure.savefig(OUTPUTFOLDER / "vxd_sections_analysis.png")
