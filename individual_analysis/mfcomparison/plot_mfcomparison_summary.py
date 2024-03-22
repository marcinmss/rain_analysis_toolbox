from pandas import read_csv
from matplotlib.pyplot import figure
import numpy as np
from pathlib import Path

from individual_analysis.analysis_variables import FIGURESPECS, LEGENDFIGURESPECS

OUTPUTFOLDER = Path(__file__).parent / "output"
PARSIVELCSVPATH = OUTPUTFOLDER / "parsivel_mfanalysis.csv"
PARSIVELCSVPATHFLUC = OUTPUTFOLDER / "parsivel_mfanalysis_with_fluctuations.csv"
STEREOCSVPATHFLUC = OUTPUTFOLDER / "stereo_mfanalysis_with_fluctuations.csv"
STEREOCSVPATH = OUTPUTFOLDER / "stereo_mfanalysis.csv"


def plot_identety(ax):
    maximum_value = max(max(ax.get_xbound()), max(ax.get_ybound()))
    ax.plot((0, maximum_value), (0, maximum_value), zorder=0)


if __name__ == "__main__":

    # Read the data frames
    stereo_data = {
        "direct_field": read_csv(
            STEREOCSVPATH,
            index_col=0,
        ),
        "fluctuations": read_csv(
            STEREOCSVPATHFLUC,
            index_col=0,
        ),
    }

    parsivel_data = {
        "direct_field": read_csv(
            PARSIVELCSVPATH,
            index_col=0,
        ),
        "fluctuations": read_csv(
            PARSIVELCSVPATHFLUC,
            index_col=0,
        ),
    }

    # Plot all of the fields
    variables = [
        ("df", "direct_field", "Fractal dimension"),
        ("alpha", "fluctuations", r"$\alpha$"),
        ("c1", "fluctuations", r"$C_1$"),
        # ("beta", "direct_field"),
        ("h", "direct_field", r"$H$"),
        ("r_square", "direct_field", r"$R^2$"),
        # ("avg_rr", "direct_field"),
        # ("total_depth", "direct_field"),
        # ("dm", "direct_field"),
    ]

    n_events = stereo_data["direct_field"].shape[0] - 1
    x = np.arange(0, n_events) + 1

    for idx, (column, flag, ax_title) in enumerate(variables, 1):
        fig = figure(**FIGURESPECS)
        ax = fig.add_subplot()

        x = np.array(stereo_data[flag][column][1:])

        # Plot the graph
        y = np.array(parsivel_data[flag][column][1:])
        ax.scatter(x, y, label="Events", c="black", s=10.0, marker="x", zorder=10)

        # Customize the axis
        ax.set_title(ax_title)
        ax.set_ylabel("Parsivel")
        ax.set_xlabel("3D Stereo")
        ax.set_ybound(min(ax.get_ybound()[0], -0.05), ax.get_ybound()[1] * 1.1)
        plot_identety(ax)

        # Plot the line for the average ensemble line
        print(f"{column}({flag}):")
        height = stereo_data[flag].loc["event_00", column]
        xmin, xmax = ax.get_xbound()
        print(f"    stereo:{height:.3f}")
        if height != 0:
            ax.vlines(
                height,
                xmin,
                xmax,
                label="Ensemble Stereo",
                linestyle="--",
                colors="dodgerblue",
                zorder=5,
                alpha=0.7,
            )

        # Plot the line for the average ensemble line
        height = parsivel_data[flag].loc["event_00", column]
        print(f"    parsivel:{height:.3f}")
        ymin, ymax = ax.get_ybound()
        if height != 0:
            ax.hlines(
                height,
                ymin,
                ymax,
                label="Ensemble Parsivel",
                linestyle="--",
                colors="orangered",
                zorder=5,
                alpha=0.7,
            )

        fig.savefig(OUTPUTFOLDER / f"summary_{column}.png")
        handles, labels = ax.get_legend_handles_labels()

        if idx > 0:
            fig = figure(**FIGURESPECS)
            ax = fig.add_subplot()
            ax.set_axis_off()
            ax.legend(handles, labels, **LEGENDFIGURESPECS)
            fig.savefig(OUTPUTFOLDER / "summary_labels.png")
