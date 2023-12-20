from stereo3d import stereo_read_from_pickle
from pathlib import Path
from pandas import DataFrame
from matplotlib import pyplot as plt
from collections import namedtuple
from multifractal_analysis import (
    spectral_analysis,
    dtm_analysis,
    fractal_dimension_analysis,
    tm_analysis,
    empirical_k_of_q,
)
from multifractal_analysis.data_prep import prep_data_ensemble
from multifractal_analysis.general import closest_smaller_power_of_2
from numpy import concatenate

MFAnalysis = namedtuple("MFAnalysis", ["df", "sa", "tm", "dtm"])

output_folder = Path(
    "/home/marcio/stage_project/mytoolbox/analysis/sprint02_and03/output/"
)
parsivel_events_folder = Path(
    "/home/marcio/stage_project/individual_analysis/sprint02/saved_events/parsivel"
)
stereo_events_folder = Path(
    "/home/marcio/stage_project/individual_analysis/sprint02/saved_events/stereo"
)


# Read the data for each events
stereo_events = [
    stereo_read_from_pickle(file_path) for file_path in stereo_events_folder.iterdir()
]

# Define the scalling regime
minimum_resolution = 0.001
size_array = closest_smaller_power_of_2(int(30 * 60 / minimum_resolution))

########################################################################################
###################### PLOT WITHOUT FLUCTUATIONS #######################################
########################################################################################
n_cols = 1
n_rows = len(stereo_events) + 1
figure = plt.figure()
figure.set_dpi(200)
a = 8
b = 6
figure.set_size_inches(a * n_cols, b * n_rows)

# Prepare the data for every event, keeping the first one as an ensemble
preped_data = [
    prep_data_ensemble(event.rain_rate(minimum_resolution), size_array)
    for event in stereo_events
]
preped_data = [concatenate(preped_data, axis=1)] + preped_data

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
    # Plot the rain rate for the event
    ax = figure.add_subplot(n_rows, n_cols, axis_count)
    if event_idx == -1:
        ax.set_axis_off()
    else:
        stereo_events[event_idx].plot_rain_rate(ax)
    axis_count += 1

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

    # Append the results to the results list
    if event_idx == -1:
        results.append(
            [
                "ensemble_events",
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
        results.append(
            [
                f"event_{event_idx+1:>02}",
                fd.df,
                dtm.alpha,
                dtm.c1,
                sa.beta,
                sa.h,
                tm.r_square,
                stereo_events[event_idx].avg_rain_rate,
                stereo_events[event_idx].total_depth_for_event,
                stereo_events[event_idx].mean_diameter_for_event,
                stereo_events[event_idx].percentage_zeros,
            ]
        )
    break

# Save the plot and the results
results = DataFrame(results, columns=columns_labels).set_index("event")
results.to_csv(output_folder / "analysis_bellow30s_03.csv")
figure.savefig(output_folder / "analysis_bellow30s_03.png")

########################################################################################
######################### PLOT WITH FLUCTUATIONS #######################################
########################################################################################

# Define the figure
figure = plt.figure()
figure.set_dpi(200)
a = 8
b = 6
figure.set_size_inches(a * n_cols, b * n_rows)

# Prepare the data for every event, keeping the first one as an ensemble
preped_data = [
    prep_data_ensemble(event.rain_rate(minimum_resolution), size_array, fluc=True)
    for event in stereo_events
]
preped_data = [concatenate(preped_data, axis=1)] + preped_data

# Run the analysis for every single event
results = []
axis_count = 1
for event_idx, data in enumerate(preped_data, -1):
    # Plot the rain rate for the event
    ax = figure.add_subplot(n_rows, n_cols, axis_count)
    if event_idx == -1:
        ax.set_axis_off()
    else:
        ax.plot(data.flatten())
    axis_count += 1

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

    # Append the results to the results list
    if event_idx == -1:
        results.append(
            [
                f"ensemble_events",
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
        results.append(
            [
                f"event_{event_idx + 1:>02}",
                fd.df,
                dtm.alpha,
                dtm.c1,
                sa.beta,
                sa.h,
                tm.r_square,
                stereo_events[event_idx].avg_rain_rate,
                stereo_events[event_idx].total_depth_for_event,
                stereo_events[event_idx].mean_diameter_for_event,
                stereo_events[event_idx].percentage_zeros,
            ]
        )
    break

# Save the plot and the results
results = DataFrame(results, columns=columns_labels).set_index("event")
results.to_csv(output_folder / "analysis_fluctuations_bellow30s_03.csv")
figure.savefig(output_folder / "analysis_fluctuations_bellow30s_03.png")
