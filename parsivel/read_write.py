from typing import Tuple
from parsivel.dataclass import ParsivelTimeSeries, ParsivelTimeStep, PARSIVELBASEAREA
from pathlib import Path
from zipfile import ZipFile
from numpy import array, ndarray
from aux_funcs.aux_datetime import (
    standard_to_dtime,
    dt_to_tstamp,
    range_dtime_1d,
    range_dtime_30s,
)

from aux_funcs.parse_filenames import construct_file_name, get_parser
import pickle

###############################################################################
################# FOR READING THE PLAIN RAW FILES #############################
###############################################################################
"""
Read from a single Parsivel .txt file
"""


def read_file(parsivel_file: str | Path) -> Tuple[float, float, ndarray]:
    file_path = Path(parsivel_file)

    # Take the lines of the file using the proper linebreak
    with open(file_path, "r") as f:
        lines = f.read().split(sep="\\r\\n")

    # Line 2 contains the precipitation rate
    precipitation_rate = float(lines[1][3:])

    # Line 3 contains the temperature
    temperature = float(lines[12][3:])

    # Line 42 or 41 contains the drop distribution matrix
    str_items = lines[40][3:-1].split(";")
    if len(str_items) != 32 * 32:
        str_items = lines[41][3:-1].split(";")

    distribution_matrix = array([int(i) for i in str_items]).reshape((32, 32))

    return (precipitation_rate, temperature, distribution_matrix)


"""
Read from a folder of ziped files for each day with the standard format.
"""


def parsivel_read_from_zips(
    beg: int, end: int, source_folder: str | Path
) -> ParsivelTimeSeries:
    # Checks if the source folder is there
    source_folder = Path(source_folder)
    assert source_folder.is_dir(), "Source folder not found."

    # Creates the storage folder and if there allread exists one delete all
    # the files inside
    temporary_storage_folder = source_folder.parent / "temporary_storage/"
    if temporary_storage_folder.exists():
        for f in temporary_storage_folder.iterdir():
            f.unlink()
    else:
        temporary_storage_folder.mkdir(exist_ok=True)

    # Get the parser for the Unziped file
    parser = get_parser(next(source_folder.glob("*.zip")).name)

    # Parse the beggingig and end dates provided
    t0, tf = standard_to_dtime(beg), standard_to_dtime(end)

    # Unzip the files and dump them all in a temporary storage file
    for dtime in range_dtime_1d(t0, tf):
        file_name = construct_file_name(dtime, parser)
        # Checks if the given day exists in the source folder
        curr_day_file = source_folder / file_name

        if curr_day_file.exists():
            # Unzips the day and puts all the files in the storage folder
            ZipFile(curr_day_file).extractall(temporary_storage_folder)

    # Read all the files and put them into the object
    missing_time_steps = []
    time_series = []
    parser = get_parser(next(temporary_storage_folder.iterdir()).name)
    for dtime in range_dtime_30s(t0, tf):
        file_name = construct_file_name(dtime, parser)
        curr_file_path = temporary_storage_folder / file_name

        # If the current step does not exist, add it to the missing time steps
        if not curr_file_path.exists():
            missing_time_steps.append(dt_to_tstamp(dtime))
            time_series.append(ParsivelTimeStep.empty(dt_to_tstamp(dtime)))
            continue

        # Exstract information from file
        try:
            precipitation_rate, temperature, distribution_matrix = read_file(
                curr_file_path
            )

            # Filter for corrupted files
            assert not (distribution_matrix < 0).any()
            new_tstep = ParsivelTimeStep(
                dt_to_tstamp(dtime),
                precipitation_rate,
                temperature,
                distribution_matrix,
            )
            if (new_tstep.matrix >= 0).all() and new_tstep.volume_mm3 < 4500:
                time_series.append(new_tstep)
            else:
                missing_time_steps.append(dt_to_tstamp(dtime))
                time_series.append(ParsivelTimeStep.empty(dt_to_tstamp(dtime)))

        except Exception:
            missing_time_steps.append(dt_to_tstamp(dtime))
            time_series.append(ParsivelTimeStep.empty(dt_to_tstamp(dtime)))

    # Clears the temporary folder and deletes its
    for f in temporary_storage_folder.iterdir():
        f.unlink()
    temporary_storage_folder.rmdir()

    return ParsivelTimeSeries(
        "parsivel",
        (dt_to_tstamp(t0), dt_to_tstamp(tf)),
        missing_time_steps,
        time_series,
        30,
        PARSIVELBASEAREA,
    )


###############################################################################
################# FOR READING AND WRITING INTO THE PICLE FORMAT ###############
###############################################################################


def write_to_picle(file_path: str | Path, series: ParsivelTimeSeries):
    file_path = Path(file_path)
    assert file_path.parent.exists(), "The file path is invalid!"
    with open(file_path, "wb") as fh:
        pickle.dump(series, fh)


def parsivel_read_from_pickle(file_path: str | Path) -> ParsivelTimeSeries:
    file_path = Path(file_path)
    assert file_path.exists(), "File doesn't exists!!!"
    with open(file_path, "rb") as fh:
        return pickle.load(fh)
