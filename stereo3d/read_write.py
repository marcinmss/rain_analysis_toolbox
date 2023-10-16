from pathlib import Path
from typing import Any, Generator, List

from numpy import array
from stereo3d.stereo3d_dataclass import Stereo3DSeries, Stereo3DRow
from csv import reader
from zipfile import ZipFile

from aux_funcs.aux_datetime import (
    round_startfinish_to_30s,
    standard_to_dtime,
    dt_to_tstamp,
    range_dtime_1d,
)


from aux_funcs.parse_filenames import construct_file_name, get_parser
import pickle

###############################################################################
################# FOR READING THE PLAIN RAW FILES #############################
###############################################################################

"""
Function for reading from a single file
"""


def read_file(file_path: str | Path) -> Generator[Stereo3DRow, Any, Any]:
    # Checks if the file exists
    file_path = Path(file_path)
    assert file_path.exists(), "The file selected does not exists"

    # Read the rows from the file
    with open(file_path, "r") as fh:
        csv_reader = reader(fh, delimiter=";")

        for row in csv_reader:
            time_stamp = int(float(row[1]) // 1000)
            timestamp_ms = float(row[1]) / 1000
            diameter = float(row[4])
            velocity = float(row[5])
            distance = float(row[6])
            shape = float(row[7])
            yield Stereo3DRow(
                time_stamp, timestamp_ms, diameter, velocity, distance, shape
            )


"""
Function for reading the file between two periods from a source folder
"""


def stereo3d_read_from_zips(
    beg: int, end: int, source_folder: str | Path
) -> Stereo3DSeries:
    start, finish = standard_to_dtime(beg), standard_to_dtime(end)
    start, finish = round_startfinish_to_30s(start, finish)

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

    # Unzip the files and dump them all in a temporary storage file
    for dtime in range_dtime_1d(start, finish):
        file_name = construct_file_name(dtime, parser)
        # Checks if the given day exists in the source folder
        curr_day_file = source_folder / file_name
        assert curr_day_file.exists(), f"Day {file_name} not found in folder."

        if curr_day_file.exists():
            # Unzips the day and puts all the files in the storage folder
            ZipFile(curr_day_file).extractall(temporary_storage_folder)

    # Read all the files and put them into the object
    first_file_name = next(temporary_storage_folder.iterdir()).name
    parser = get_parser(first_file_name)
    series: List[Stereo3DRow] = []
    for dtime in range_dtime_1d(start, finish):
        file_name = construct_file_name(dtime, parser)
        curr_file_path = temporary_storage_folder / file_name

        if not curr_file_path.exists():
            continue

        series.extend(read_file(curr_file_path))

    # Clears the temporary folder and deletes its
    for f in temporary_storage_folder.iterdir():
        f.unlink()
    temporary_storage_folder.rmdir()

    # Filter for the objects that are in the period requested
    tstamp0, tstampf = dt_to_tstamp(start), dt_to_tstamp(finish)
    series = [item for item in series if tstamp0 < item.timestamp < tstampf]

    return Stereo3DSeries((tstamp0, tstampf), array(series))


###############################################################################
################# FOR READING AND WRITING INTO THE PICLE FORMAT ###############
###############################################################################


def write_to_picle(file_path: str | Path, series: Stereo3DSeries):
    file_path = Path(file_path)
    assert file_path.parent.exists(), "The file path is invalid!"
    with open(file_path, "wb") as fh:
        pickle.dump(series, fh)


def stereo_read_from_pickle(file_path: str | Path) -> Stereo3DSeries:
    file_path = Path(file_path)
    assert file_path.exists(), "File doesn't exists!!!"
    with open(file_path, "rb") as fh:
        return pickle.load(fh)
