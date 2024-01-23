from collections import namedtuple
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUTFOlDER = Path(__file__).parent / "output"

WorkAround = namedtuple("WorkAround", ["variable_name", "correct_data"])


def plot_identety(ax):
    maximum_value = max(max(ax.get_xbound()), max(ax.get_ybound()))
    ax.plot((0, maximum_value), (0, maximum_value), color="black")


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

    n_cols = 5
    n_rows = 1

    figure = plt.figure()
    figure.set_dpi(200)
    figure.set_size_inches((n_cols * 5, n_rows * 4))

    for idx, (column, flag) in enumerate(variables, 1):
        ax = figure.add_subplot(n_rows, n_cols, idx)

        y = np.array(stereo_data[flag][column][1:])

        # Plot the line for the average ensemble line
        print(f"{column}({flag}):")
        height = stereo_data[flag].loc["ensemble_of_events", column]
        print(f"    stereo:{height:.3f}")
        # if height != 0:
        #     ax.hlines(
        #         height,
        #         1,
        #         n_events,
        #         label="Ensemble Stereo",
        #         linestyle="--",
        #         colors="dodgerblue",
        #     )

        x = np.array(parsivel_data[flag][column][1:])

        # Plot the line for the average ensemble line
        height = parsivel_data[flag].loc["ensemble_of_events", column]
        print(f"    parsivel:{height:.3f}")
        # if height != 0:
        #     ax.hlines(
        #         height,
        #         1,
        #         n_events,
        #         label="Ensemble Parsivel",
        #         linestyle="--",
        #         colors="orangered",
        #     )

        # Plot the graph
        ax.scatter(x, y, label="Parsivel", c="orangered", s=5.0, marker="x")

        # Customize the axis
        ax.set_title(column.upper())
        plot_identety(ax)

    figure.savefig(OUTPUTFOlDER / "overall_comparison.png", bbox_inches="tight")


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
