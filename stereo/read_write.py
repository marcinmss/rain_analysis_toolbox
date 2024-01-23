from pathlib import Path
from typing import Any, Generator, List, Tuple
from pandas import DataFrame

from numpy import array, ndarray
from stereo.dataclass import MAXDIST, MINDIST, Stereo3DSeries, Stereo3DRow
from csv import reader
from zipfile import ZipFile
from pyarrow import schema, field as pa_field, table as patable
from pyarrow.parquet import write_table, read_table

from aux_funcs.aux_datetime import (
    round_startfinish_to_30s,
    standard_to_dtime,
    dt_to_tstamp,
    range_dtime_1d,
)


from aux_funcs.parse_filenames import construct_file_name, get_parser
import pickle

"""
There is an ofset of two hours between the parsivel data and the stereo3d data.
Assuming that the parsivel is correct, i am going to adjust the timestamps in 
the stereo3d by the diference (2 hours).
I bellive that this exists because the parsivel is in Paris time and the stereo3d
is in utc.
"""
OFFSETSECONDS = 3600 * 2

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
            time_stamp = int(float(row[1]) // 1000) - OFFSETSECONDS
            timestamp_ms = float(row[1]) / 1000 - OFFSETSECONDS
            diameter = float(row[4])
            velocity = float(row[5])
            distance = float(row[6])
            shape = float(row[7])
            if diameter < 25:
                yield Stereo3DRow(
                    time_stamp, timestamp_ms, diameter, velocity, distance, shape
                )


"""
Function for reading the file between two periods from a source folder
"""


def stereo_read_from_zips(
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
        # assert curr_day_file.exists(), f"Day {file_name} not found in folder."

        if curr_day_file.exists():
            # Unzips the day and puts all the files in the storage folder
            ZipFile(curr_day_file).extractall(temporary_storage_folder)

    # Read all the files and put them into the object
    assert (
        sum(1 for _ in temporary_storage_folder.iterdir()) > 0
    ), "No file was found in this time period"
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

    return Stereo3DSeries(
        "stereo3d", (tstamp0, tstampf), array(series), (MINDIST, MAXDIST)
    )


###############################################################################
################# FOR READING AND WRITING INTO THE PICLE FORMAT ###############
###############################################################################


drop_fields = {
    "timestamp": "int64",
    "timestamp_ms": "float64",
    "diameter": "float64",
    "velocity": "float64",
    "distance_to_sensor": "float64",
    "shape_parameter": "float64",
}


def write_to_parquet(file_path: str | Path, series: Stereo3DSeries):
    # Set up the table to reseave data
    table = patable(
        DataFrame(
            data=[
                [drop.__getattribute__(column) for column in drop_fields.keys()]
                for drop in series
            ],
            columns=[field for field in drop_fields.keys()],
        )
    )

    # Add the metadata for series
    series_metadata = {
        "device": series.device,
        "duration_beg": str(series.duration[0]),
        "duration_end": str(series.duration[1]),
        "limits_area_of_study_beg": str(series.limits_area_of_study[0]),
        "limits_area_of_study_end": str(series.limits_area_of_study[1]),
    }

    my_schema = schema(
        [pa_field(field, dtype) for field, dtype in drop_fields.items()],
        metadata=series_metadata,
    )

    table = table.cast(my_schema)

    file_path = Path(file_path)
    assert file_path.parent.exists(), "The file path is invalid!"
    write_table(table, file_path)


def stereo_read_from_parquet(file_path: str | Path) -> Stereo3DSeries:
    file_path = Path(file_path)
    assert file_path.exists(), "File doesn't exists!!!"

    print("Read table")
    table = read_table(file_path)

    metadata = table.schema.metadata
    duration = (int(metadata[b"duration_beg"]), int(metadata[b"duration_end"]))
    limits_area_of_study = (
        float(metadata[b"limits_area_of_study_beg"]),
        float(metadata[b"limits_area_of_study_end"]),
    )
    devie = str(metadata[b"device"])

    print("Read the Metadata")

    print("Converted to pandas")
    table = table.to_pandas()

    series = array([Stereo3DRow(**row[1]) for row in table.iterrows()])
    return Stereo3DSeries(
        devie,
        duration,
        series,
        limits_area_of_study,
    )


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
