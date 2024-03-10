from collections import namedtuple
from pandas import read_csv
from matplotlib.pyplot import figure
import numpy as np
from pathlib import Path

OUTPUTFOlDER = Path(__file__).parent / "output"

WorkAround = namedtuple("WorkAround", ["variable_name", "correct_data"])


def overall_comparison(
    parsivel_csv_path: Path,
    stereo_csv_path: Path,
    parsivel_fluctuations_csv_path: Path,
    stereo_fluctuations_csv_path: Path,
):
    # Read the data frames
    stereo_data = {
        "direct_field": read_csv(
            stereo_csv_path,
            index_col=0,
        ),
        "fluctuations": read_csv(
            stereo_fluctuations_csv_path,
            index_col=0,
        ),
    }

    parsivel_data = {
        "direct_field": read_csv(
            parsivel_csv_path,
            index_col=0,
        ),
        "fluctuations": read_csv(
            parsivel_fluctuations_csv_path,
            index_col=0,
        ),
    }

    # Plot all of the fields
    variables = [
        ("df", "direct_field"),
        ("alpha", "fluctuations"),
        ("c1", "fluctuations"),
        # ("beta", "direct_field"),
        ("h", "direct_field"),
        ("r_square", "direct_field"),
        # ("avg_rr", "direct_field"),
        # ("total_depth", "direct_field"),
        # ("dm", "direct_field"),
    ]

    n_events = stereo_data["direct_field"].shape[0] - 1
    x = np.arange(0, n_events) + 1
    handles, labels = "",""
    for idx, (column, flag) in enumerate(variables, 1):
        fig = figure(dpi=300,figsize=(5, 4), layout="constrained")
        ax = fig.add_subplot(1,1,1)

        y = np.array(stereo_data[flag][column][1:])

        # Plot the line for the average ensemble line
        print(f"{column}({flag}):")
        height = stereo_data[flag].loc["ensemble_of_events", column]
        print(f"    stereo:{height:.3f}")
        if height != 0:
            ax.hlines(
                height,
                1,
                n_events,
                label="Ensemble Stereo",
                linestyle="--",
                colors="dodgerblue",
            )

        # Plot the graph
        ax.scatter(x, y, label="Stereo", c="dodgerblue", s=5.0)

        y = np.array(parsivel_data[flag][column][1:])

        # Plot the line for the average ensemble line
        height = parsivel_data[flag].loc["ensemble_of_events", column]
        print(f"    parsivel:{height:.3f}")
        if height != 0:
            ax.hlines(
                height,
                1,
                n_events,
                label="Ensemble Parsivel",
                linestyle="--",
                colors="orangered",
            )

        # Plot the graph
        ax.scatter(x, y, label="Parsivel", c="orangered", s=5.0)

        # Customize the axis
        ax.set_title(column.upper())
        ax.set_ybound(min(ax.get_ybound()[0], -0.05), ax.get_ybound()[1] * 1.1)

        fig.savefig(OUTPUTFOlDER / f"summary_{column}.png")
        handles, labels = ax.get_legend_handles_labels()

    # Plot the legend
    fig = figure(dpi=300,figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1,1,1)
    ax.set_axis_off()
    fig.legend(handles, labels, loc="center", fontsize=16, frameon=False)
    fig.savefig(OUTPUTFOlDER / f"summary_labels.png")



if __name__ == "__main__":
    parsivel_csv_path = OUTPUTFOlDER / "parsivel_mfanalysis.csv"
    parsivel_fluctuations_csv_path = (
        OUTPUTFOlDER / "parsivel_mfanalysis_with_fluctuations.csv"
    )
    stereo_fluctuations_csv_path = (
        OUTPUTFOlDER / "stereo_mfanalysis_with_fluctuations.csv"
    )
    stereo_csv_path = OUTPUTFOlDER / "stereo_mfanalysis.csv"
    overall_comparison(
        parsivel_csv_path,
        stereo_csv_path,
        parsivel_fluctuations_csv_path,
        stereo_fluctuations_csv_path,
    )
