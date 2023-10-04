from parsivel.parsivel_dataclass import ParsivelTimeSeries, ParsivelInfo
from pathlib import Path
from zipfile import ZipFile
from numpy import array
from aux_funcs.aux_funcs_read_files import (
    construct_file_name,
    get_parser,
    range_between_dates,
    range_between_timestamps_30s,
    start_finish_to_tstamp,
)
from datetime import datetime


"""
Read from a single Parsivel .txt file
"""


def read_file(parsivel_file: str | Path) -> ParsivelInfo:
    file_path = Path(parsivel_file)
    # Checks if the file exists
    assert file_path.is_file(), "File does not exists"

    # # Extracts the date and the time from the file
    # timestamp = int("".join(file_path.name.strip(".txt").split("_")[-2:]))

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

    distribution_matrix = array([int(i) for i in str_items]).reshape((32, 32)).T

    return ParsivelInfo(0, precipitation_rate, temperature, distribution_matrix)


"""
Read from a folder of ziped files for each day with the standard format.
"""


def read_from_source(
    beg: int, end: int, source_folder: str | Path
) -> ParsivelTimeSeries:
    # Checks if the source folder is there, and creates the storage folder
    source_folder = Path(source_folder)
    assert source_folder.is_dir(), "Source folder not found."

    #
    temporary_storage_folder = source_folder.parent / "temporary_storage/"
    temporary_storage_folder.mkdir(exist_ok=True)

    # Get the parser for the Unziped file
    first_file_name = next(source_folder.glob("*.zip")).name
    parser = get_parser(first_file_name)
    file_names = [
        construct_file_name(date, parser) for date in range_between_dates(beg, end)
    ]
    assert len(file_names) > 0, "Time intervals are not correct."

    # Unzip the files and dump them all in a temporary storage file
    for file_name in file_names:
        # Checks if the given day exists in the source folder
        curr_day_file = source_folder / file_name
        assert curr_day_file.exists(), f"Day {file_name} not found in folder."

        # Unzips the day and puts all the files in the storage folder
        ZipFile(curr_day_file).extractall(temporary_storage_folder)

    # Read all the files and put them into the object
    missing_time_steps = []
    time_series = []
    first_file_name = next(temporary_storage_folder.iterdir()).name
    parser = get_parser(first_file_name)
    for date in range_between_timestamps_30s(beg, end):
        file_name = construct_file_name(date, parser)
        curr_file_path = temporary_storage_folder / file_name

        if not curr_file_path.exists():
            missing_time_steps.append(int(date.timestamp()))
            time_series.append(ParsivelInfo.empty(int(date.timestamp())))
            continue

        # TODO: Make so that i dont have to create this object and then reasing it
        pars_obj = read_file(curr_file_path)
        pars_obj.timestamp = int(date.timestamp())
        time_series.append(pars_obj)

    # Clears the temporary folder and deletes its
    for f in temporary_storage_folder.iterdir():
        f.unlink()
    temporary_storage_folder.rmdir()

    start = start_finish_to_tstamp(datetime.strptime(str(beg), "%Y%m%d%H%M%S"))
    finish = start_finish_to_tstamp(datetime.strptime(str(end), "%Y%m%d%H%M%S"))

    return ParsivelTimeSeries((start, finish), missing_time_steps, time_series, 30)
