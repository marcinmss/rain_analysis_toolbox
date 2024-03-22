from individual_analysis.analysis_variables import FIGURESPECS
from parsivel import parsivel_read_from_pickle
from pathlib import Path
from matplotlib.pyplot import figure
from multifractal_analysis import (
    spectral_analysis,
    dtm_analysis,
    fractal_dimension_analysis,
    tm_analysis,
    empirical_k_of_q,
)
from multifractal_analysis.data_prep import prep_data_ensemble
from numpy import ndarray, concatenate


from stereo.read_write import stereo_read_from_pickle


OUTPUTFOLDER = Path(__file__).parent / "output/"
PARSIVELEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
)
STEREOEVENTSFOLDER = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo/"
)


def plot_mfcomparison_graphs(
    field: ndarray, output_folder: Path, device: str, fluctuations: bool
):
    field_type = "fluc" if fluctuations is True else "df"

    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    spectral_analysis(field, ax)
    opeartion = "sa"
    ax.set_title("")
    fig.savefig(output_folder / f"{device}_{field_type}_{opeartion}.png")

    # Run fractal dimension analysis and plot the graph
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    fractal_dimension_analysis(field, ax)
    opeartion = "fd"
    ax.set_title("")
    fig.savefig(output_folder / f"{device}_{field_type}_{opeartion}.png")

    # Run trace moment analysis and plot the graph
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    tm_analysis(field, ax)
    opeartion = "tm"
    ax.set_title("")
    fig.savefig(output_folder / f"{device}_{field_type}_{opeartion}.png")

    # Run Double trace moment analysis and plot the graphs
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    dtm_analysis(field, ax)
    opeartion = "dtm"
    ax.set_title("")
    fig.savefig(output_folder / f"{device}_{field_type}_{opeartion}.png")

    # Plot the empirical k of q
    fig = figure(**FIGURESPECS)
    ax = fig.add_subplot()
    empirical_k_of_q(field, ax)
    opeartion = "kofq"
    ax.set_title("")
    fig.savefig(output_folder / f"{device}_{field_type}_{opeartion}.png")


if __name__ == "__main__":
    print("Reading The Events for Parsivel.")
    parsivel_rain_rate = [
        parsivel_read_from_pickle(file_path).rain_rate()
        for file_path in PARSIVELEVENTSFOLDER.iterdir()
    ]
    # Prepare the data for every event, keeping the first one as an ensemble
    print("Preparing the data for the direct field")
    preped_data_df = concatenate(
        [prep_data_ensemble(event, 2**6, fluc=False) for event in parsivel_rain_rate],
        axis=1,
    )

    print("Preparing the data for the fluctuations")
    preped_data_fluc = concatenate(
        [prep_data_ensemble(event, 2**6, fluc=True) for event in parsivel_rain_rate],
        axis=1,
    )

    del parsivel_rain_rate

    print("Running Analysis for direct field.")
    plot_mfcomparison_graphs(preped_data_df, OUTPUTFOLDER, "parsivel", False)

    print("Running Analysis for fluctuations.")
    plot_mfcomparison_graphs(preped_data_fluc, OUTPUTFOLDER, "parsivel", True)

    print("Reading The Events for Stereo.")
    stereo_rain_rate = [
        stereo_read_from_pickle(file_path).rain_rate()
        for file_path in STEREOEVENTSFOLDER.iterdir()
    ]

    # Prepare the data for every event, keeping the first one as an ensemble
    print("Preparing the data for the direct field")
    preped_data_df = concatenate(
        [prep_data_ensemble(event, 2**6, fluc=False) for event in stereo_rain_rate],
        axis=1,
    )

    print("Preparing the data for the fluctuations")
    preped_data_fluc = concatenate(
        [prep_data_ensemble(event, 2**6, fluc=True) for event in stereo_rain_rate],
        axis=1,
    )

    del stereo_rain_rate

    print("Running Analysis for direct field.")
    plot_mfcomparison_graphs(preped_data_df, OUTPUTFOLDER, "stereo", False)

    print("Running Analysis for fluctuations.")
    plot_mfcomparison_graphs(preped_data_fluc, OUTPUTFOLDER, "stereo", True)
