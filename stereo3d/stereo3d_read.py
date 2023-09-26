from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List

from numpy import array
from stereo3d.stereo3d_dataclass import Stereo3DSeries, Stereo3DRow
from csv import reader
from zipfile import ZipFile
from aux_funcs.aux_funcs_read_files import (
    range_between_dates,
    get_parser,
    construct_file_name,
)

OFFSET = 2 * 3600

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


def start_finish_to_timestamp(start_finish: datetime) -> int:
    start_finish = start_finish.replace(second=start_finish.second // 30 * 30)
    return int(start_finish.timestamp())


def read_from_source(beg: int, end: int, source_folder: str | Path) -> Stereo3DSeries:
    # Checks if the source folder is there
    source_folder = Path(source_folder)
    assert source_folder.is_dir(), "Source folder not found."

    # Creates a temporary storage folder for dumping the unziped files
    temporary_storage_folder = source_folder.parent / "temporary_storage/"
    temporary_storage_folder.mkdir(exist_ok=True)
    if any(temporary_storage_folder.iterdir()):
        pass
        # TODO: Delete all files in that folder

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
    first_file_name = next(temporary_storage_folder.iterdir()).name
    parser = get_parser(first_file_name)
    series: List[Stereo3DRow] = []
    for date in range_between_dates(beg, end):
        file_name = construct_file_name(date, parser)
        curr_file_path = temporary_storage_folder / file_name

        if not curr_file_path.exists():
            continue

        series.extend(read_file(curr_file_path))

    # Clears the temporary folder and deletes its
    for f in temporary_storage_folder.iterdir():
        f.unlink()
    temporary_storage_folder.rmdir()

    # Filter for the objects that are in the period requested
    start = (
        start_finish_to_timestamp(datetime.strptime(str(beg), "%Y%m%d%H%M%S")) + OFFSET
    )

    finish = (
        start_finish_to_timestamp(datetime.strptime(str(end), "%Y%m%d%H%M%S")) + OFFSET
    )
    series = [item for item in series if start < item.timestamp < finish]

    return Stereo3DSeries((start, finish), array(series))
