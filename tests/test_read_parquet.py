from pathlib import Path
from pandas import DataFrame, array
from pyarrow.parquet import read_table
from stereo.dataclass import Stereo3DRow, Stereo3DSeries
from stereo import stereo_read_from_pickle
from time import time

file_path = Path("/home/marcio/stage_project/mytoolbox/tests/test.parquet")


def method1():
    t0 = time()
    obj = stereo_read_from_pickle(
        "/home/marcio/stage_project/data/saved_events/Set01/stereo_full_event.obj"
    )
    print(f"O tempo foi {time() - t0:.2f}")
    return obj


def method2():
    table = read_table(file_path)

    t0 = time()
    metadata = table.schema.metadata
    table = read_table(file_path)

    metadata = table.schema.metadata
    duration = (int(metadata[b"duration_beg"]), int(metadata[b"duration_end"]))
    limits_area_of_study = (
        float(metadata[b"limits_area_of_study_beg"]),
        float(metadata[b"limits_area_of_study_end"]),
    )
    devie = str(metadata[b"device"])

    series = array(
        [
            Stereo3DRow(
                row.timestamp,
                row.timestamp_ms,
                row.diameter,
                row.velocity,
                row.distance_to_sensor,
                row.shape_parameter,
            )
            for row in table.to_pandas().itertuples()
        ]
    )
    print(f"O tempo foi {time() - t0:.2f}")

    Stereo3DSeries(
        devie,
        duration,
        series,
        limits_area_of_study,
    )


def method3():
    table = read_table(file_path)
    t0 = time()
    metadata = table.schema.metadata
    table = read_table(file_path)

    metadata = table.schema.metadata
    duration = (int(metadata[b"duration_beg"]), int(metadata[b"duration_end"]))
    limits_area_of_study = (
        float(metadata[b"limits_area_of_study_beg"]),
        float(metadata[b"limits_area_of_study_end"]),
    )
    devie = str(metadata[b"device"])
    table = table.to_pandas()
    series = array(
        [
            Stereo3DRow(
                timestamp,
                timestamp_ms,
                diameter,
                velocity,
                distance_to_sensor,
                shape_parameter,
            )
            for (
                timestamp,
                timestamp_ms,
                diameter,
                velocity,
                distance_to_sensor,
                shape_parameter,
            ) in zip(
                table["timestamp"],
                table["timestamp_ms"],
                table["diameter"],
                table["velocity"],
                table["distance_to_sensor"],
                table["shape_parameter"],
            )
        ]
    )

    print(f"O tempo foi {time() - t0:.2f}")
    Stereo3DSeries(
        devie,
        duration,
        series,
        limits_area_of_study,
    )


if __name__ == "__main__":
    print("Method 01")
    method1()
    print("Method 02")
    method2()
    print("Method 03")
    method3()
