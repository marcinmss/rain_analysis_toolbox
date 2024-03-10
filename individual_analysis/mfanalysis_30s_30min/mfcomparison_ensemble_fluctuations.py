from typing import List
from parsivel import parsivel_read_from_pickle
from pathlib import Path
from pandas import DataFrame
from matplotlib.pyplot import figure
from multifractal_analysis import (
    spectral_analysis,
    dtm_analysis,
    fractal_dimension_analysis,
    tm_analysis,
    empirical_k_of_q,
)
from multifractal_analysis.data_prep import prep_data_ensemble
from numpy import  ndarray, concatenate
from parsivel import ParsivelTimeSeries


OUTPUTFOLDER = Path(__file__).parent / "output/"
parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
)


def plot_mfcomparison_graphs(
    data: ndarray ,output_folder: Path, device:str
):
    name = ""

    fig = figure(dpi=300,figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1,1,1)
    sa = spectral_analysis(data, ax)
    fig.savefig(output_folder / f"mfcomparison_{device}_sa_{name}.png")

    # Run fractal dimension analysis and plot the graph
    fig = figure(dpi=300,figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1,1,1)
    fd = fractal_dimension_analysis(data, ax)

    # Run trace moment analysis and plot the graph
    fig = figure(dpi=300,figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1,1,1)
    tm = tm_analysis(data, ax)

    # Run Double trace moment analysis and plot the graphs
    fig = figure(dpi=300,figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1,1,1)
    dtm = dtm_analysis(data, ax)

    # Plot the empirical k of q
    fig = figure(dpi=300,figsize=(5, 4), layout="constrained")
    ax = fig.add_subplot(1,1,1)
    empirical_k_of_q(data, ax)


    fig.savefig(output_folder / f"{name}.png")


if __name__ == "__main__":
    print("Reading The Events for Parsivel.")
    events = [
        parsivel_read_from_pickle(file_path)
        for file_path in parsivel_events_folder.iterdir()
    ]
    # Prepare the data for every event, keeping the first one as an ensemble
    preped_data = concatenate([
        prep_data_ensemble(event.rain_rate(), 2**6, fluc= True)
        for event in events
    ], axis=1)
    print("    Running Analysis for direct field.")
    plot_mfcomparison_graphs(preped_data, OUTPUTFOLDER, "parsivel")

    print("    Running Analysis for fluctuations")
    plot_mfcomparison_graphs(preped_data, OUTPUTFOLDER, "parsivel")
