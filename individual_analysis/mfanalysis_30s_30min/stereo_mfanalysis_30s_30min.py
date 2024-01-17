from typing import List
from stereo3d import stereo_read_from_pickle
from pathlib import Path
from pandas import DataFrame
from matplotlib import pyplot as plt
from multifractal_analysis import (
    spectral_analysis,
    dtm_analysis,
    fractal_dimension_analysis,
    tm_analysis,
    empirical_k_of_q,
)
from multifractal_analysis.data_prep import prep_data_ensemble
from numpy import concatenate

from stereo3d.stereo3d_dataclass import Stereo3DSeries


output_folder = Path(__file__).parent / "output/"
stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo/"
)


def mf_analysis_multiple_events_stereo(
    output_folder: Path, fluctuations: bool, events: List[Stereo3DSeries]
):
    # Prepare the data for every event, keeping the first one as an ensemble
    preped_data = [
        prep_data_ensemble(event.rain_rate(), 2**6, fluc=fluctuations)
        for event in events
    ]
    preped_data = [concatenate(preped_data, axis=1)] + preped_data

    # Define the figure
    n_cols = 6
    n_rows = 1
    figure = plt.figure()
    figure.set_layout_engine("tight")
    figure.set_dpi(200)
    a = 8
    b = 6
    figure.set_size_inches(a * n_cols, b * n_rows)

    # Run the analysis for every single event
    results = []
    columns_labels = [
        "event",
        "df",
        "alpha",
        "c1",
        "beta",
        "h",
        "r_square",
        "avg_rr",
        "total_depth",
        "dm",
        "percentage_zeros",
    ]
    axis_count = 1
    for event_idx, data in enumerate(preped_data, -1):
        # Append the results to the results list
        if event_idx == -1:
            # Run spectral analysis and plot the graph
            ax = figure.add_subplot(n_rows, n_cols, axis_count)
            axis_count += 1
            sa = spectral_analysis(data, ax)

            # Run fractal dimension analysis and plot the graph
            ax = figure.add_subplot(n_rows, n_cols, axis_count)
            axis_count += 1
            fd = fractal_dimension_analysis(data, ax)

            # Run trace moment analysis and plot the graph
            ax = figure.add_subplot(n_rows, n_cols, axis_count)
            axis_count += 1
            tm = tm_analysis(data, ax)

            # Run Double trace moment analysis and plot the graphs
            ax1 = figure.add_subplot(n_rows, n_cols, axis_count)
            axis_count += 1
            ax2 = figure.add_subplot(n_rows, n_cols, axis_count)
            axis_count += 1
            dtm = dtm_analysis(data, ax1, ax2)

            # Plot the empirical k of q
            ax = figure.add_subplot(n_rows, n_cols, axis_count)
            axis_count += 1
            empirical_k_of_q(data, ax)
            results.append(
                [
                    "ensemble_of_events",
                    fd.df,
                    dtm.alpha,
                    dtm.c1,
                    sa.beta,
                    sa.h,
                    tm.r_square,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        else:
            # Run spectral analysis and plot the graph
            sa = spectral_analysis(data)

            # Run fractal dimension analysis and plot the graph
            fd = fractal_dimension_analysis(data)

            # Run trace moment analysis and plot the graph
            tm = tm_analysis(data)

            # Run Double trace moment analysis and plot the graphs
            dtm = dtm_analysis(data)

            # Plot the empirical k of q
            results.append(
                [
                    f"event_{event_idx+1:>02}",
                    fd.df,
                    dtm.alpha,
                    dtm.c1,
                    sa.beta,
                    sa.h,
                    tm.r_square,
                    events[event_idx].avg_rain_rate,
                    events[event_idx].total_depth_for_event,
                    events[event_idx].mean_diameter_for_event,
                    events[event_idx].percentage_zeros,
                ]
            )

    # Save the plot and the results
    name = "stereo_mfanalysis"
    name = name + "_with_fluctuations" if fluctuations is True else name

    results = DataFrame(results, columns=columns_labels).set_index("event")
    results.to_csv(output_folder / f"{name}.csv")
    figure.savefig(output_folder / f"{name}.png")


if __name__ == "__main__":
    print("Reading The Events for 3D Stereo.")
    events = [
        stereo_read_from_pickle(file_path)
        for file_path in stereo_events_folder.iterdir()
    ]
    print("Running Analysis for direct field.")
    mf_analysis_multiple_events_stereo(output_folder, False, events)
    print("Running Analysis for the fluctuations.")
    mf_analysis_multiple_events_stereo(output_folder, True, events)
