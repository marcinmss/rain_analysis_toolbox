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
from multifractal_analysis.general import closest_smaller_power_of_2
from multifractal_analysis.data_prep import prep_data_ensemble
from numpy import concatenate

from gc import collect


output_folder = Path("/home/marcio/stage_project/individual_analysis/sprint05b/output/")
parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/parsivel/"
)
stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/stereo/"
)


# Read the data for each events
stereo_events = [
    stereo_read_from_pickle(file_path) for file_path in stereo_events_folder.iterdir()
]

# Define the scalling regime
minimum_resolution = 0.001
size_array = closest_smaller_power_of_2(int(30 * 60 / minimum_resolution))

data = concatenate(
    [
        prep_data_ensemble(event.rain_rate(minimum_resolution), size_array)
        for event in stereo_events
    ],
    axis=1,
)

print(f"Data shape {data.shape}")
print("\t Data preped.")

data.shape
print("Running Analysis for direct field.")
# Define the figure
n_cols = 6
n_rows = 1
figure = plt.figure()
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

# Run spectral analysis and plot the graph
print("\t Running Spectral Analysis.")
ax = figure.add_subplot(n_rows, n_cols, axis_count)
axis_count += 1
sa = spectral_analysis(data, ax)
print("\t Spectral Analysis Done.")
collect()

# Run fractal dimension analysis and plot the graph
print("\t Running Fractal Dimension.")
ax = figure.add_subplot(n_rows, n_cols, axis_count)
axis_count += 1
fd = fractal_dimension_analysis(data, ax)
print("\t Fractal Dimension Done.")

# # Run trace moment analysis and plot the graph
# print("\t Running TM Analysis.")
# ax = figure.add_subplot(n_rows, n_cols, axis_count)
# axis_count += 1
# tm = tm_analysis(data, ax)
# print("\t TM Analysis Done.")
# collect()
#
# # Run Double trace moment analysis and plot the graphs
# print("\t Running DTM Analysis.")
# ax1 = figure.add_subplot(n_rows, n_cols, axis_count)
# axis_count += 1
# ax2 = figure.add_subplot(n_rows, n_cols, axis_count)
# axis_count += 1
# dtm = dtm_analysis(data, ax1, ax2)
# print("\t DTM Analysis Done.")
# collect()
#
# # Plot the empirical k of q
# ax = figure.add_subplot(n_rows, n_cols, axis_count)
# axis_count += 1
# empirical_k_of_q(data, ax)
# results.append(
#     [
#         "ensemble_of_events",
#         fd.df,
#         dtm.alpha,
#         dtm.c1,
#         sa.beta,
#         sa.h,
#         tm.r_square,
#         0,
#         0,
#         0,
#         0,
#     ]
# )
# # Save the plot and the results
# name = "stereo_mfanalysis_1ms_30min"
# name = name + "_with_fluctuations" if fluctuations is False else name
#
# results = DataFrame(results, columns=columns_labels).set_index("event")
# results.to_csv(output_folder / f"{name}.csv")
# figure.savefig(output_folder / f"{name}.png")
