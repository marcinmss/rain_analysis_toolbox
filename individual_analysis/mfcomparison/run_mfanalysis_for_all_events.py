from typing import List
from parsivel.dataclass import ParsivelTimeSeries
from stereo import stereo_read_from_pickle
from pathlib import Path
from pandas import DataFrame
from parsivel.read_write import parsivel_read_from_pickle
from multifractal_analysis import (
    spectral_analysis,
    dtm_analysis,
    fractal_dimension_analysis,
    tm_analysis,
)
from multifractal_analysis.data_prep import prep_data_ensemble
from numpy import concatenate

from stereo.dataclass import Stereo3DSeries


OUTPUTFOLDER = Path(__file__).parent / "output/"
PARSIVELEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
)
STEREOEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo/"
)


def mfanalysis_multiple_events(
    device: str,
    output_folder: Path,
    fluctuations: bool,
    events: List[ParsivelTimeSeries] | List[Stereo3DSeries],
):
    # Prepare the data for every event, keeping the first one as an ensemble
    preped_data = [
        prep_data_ensemble(event.rain_rate(), 2**6, fluc=fluctuations)
        for event in events
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

    for event_idx, data in enumerate(preped_data, -1):
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
                events[event_idx].avg_rain_rate(),
                events[event_idx].total_depth_for_event(),
                events[event_idx].mean_diameter_for_event(),
                events[event_idx].percentage_zeros(),
            ]
        )

    # Save the plot and the results
    name = f"{device}_mfanalysis"
    name = name + "_with_fluctuations" if fluctuations is True else name

    results = DataFrame(results, columns=columns_labels).set_index("event")
    results.to_csv(output_folder / f"{name}.csv")


if __name__ == "__main__":

    print("1. Running analysis for Stereo.")
    print("Reading Events.")
    stereo_events = [
        stereo_read_from_pickle(file_path) for file_path in STEREOEVENTSFOLDER.iterdir()
    ]

    # Prepare the data for every event, keeping the first one as an ensemble
    print("Running Analysis for direct field.")
    mfanalysis_multiple_events("stereo", OUTPUTFOLDER, False, stereo_events)
    print("Running Analysis for fluctuations.")
    mfanalysis_multiple_events("stereo", OUTPUTFOLDER, True, stereo_events)

    del stereo_events

    print("Reading Events.")
    parsivel_events = [
        parsivel_read_from_pickle(file_path)
        for file_path in PARSIVELEVENTSFOLDER.iterdir()
    ]

    # Prepare the data for every event, keeping the first one as an ensemble
    print("Running Analysis for direct field.")
    mfanalysis_multiple_events("parsivel", OUTPUTFOLDER, False, parsivel_events)
    print("Running Analysis for fluctuations.")
    mfanalysis_multiple_events("parsivel", OUTPUTFOLDER, True, parsivel_events)

    del parsivel_events
    print("DONE.")
