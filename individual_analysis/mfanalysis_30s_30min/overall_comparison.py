from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUTFOlDER = Path(__file__).parent / "output"


def overall_comparison(parsivel_csv_path: Path, stereo_csv_path: Path):
    # Read both data frames
    stereo_data = read_csv(
        stereo_csv_path,
        index_col=0,
    )
    parsivel_data = read_csv(
        parsivel_csv_path,
        index_col=0,
    )
    n_events = parsivel_data.shape[0] - 1
    # Plot all of the fields
    fields = [
        "df",
        "alpha",
        "c1",
        "beta",
        "h",
        "r_square",
        "avg_rr",
        "total_depth",
        "dm",
    ]
    n_cols = 5
    n_rows = 2

    figure = plt.figure()
    figure.set_dpi(200)
    figure.set_size_inches((n_cols * 5, n_rows * 4))

    x = np.arange(0, n_events) + 1
    for idx, column in enumerate(fields, 1):
        ax = figure.add_subplot(n_rows, n_cols, idx)

        y = np.array(stereo_data[column][1:])

        # Plot the line for the average ensemble line
        hight = stereo_data.loc["ensemble_of_events", column]
        if hight != 0:
            ax.hlines(
                hight,
                1,
                n_events,
                label="Ensemble Stereo",
                linestyle="--",
                colors="dodgerblue",
            )

        # Plot the graph
        ax.scatter(x, y, label="Stereo", c="dodgerblue", s=5.0)

        y = np.array(parsivel_data[column][1:])

        # Plot the line for the average ensemble line
        hight = parsivel_data.loc["ensemble_of_events", column]
        if hight != 0:
            ax.hlines(
                hight,
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
        # ax.set_xticks(list(x))
        if idx == 1:
            ax.legend(frameon=False, loc=(5.0, -0.75))

    figure.savefig(OUTPUTFOlDER / "overall_comparison.png", bbox_inches="tight")


if __name__ == "__main__":
    parsivel_csv_path = OUTPUTFOlDER / "parsivel_mfanalysis.csv"
    stereo_csv_path = OUTPUTFOlDER / "stereo_mfanalysis.csv"
    overall_comparison(parsivel_csv_path, stereo_csv_path)
