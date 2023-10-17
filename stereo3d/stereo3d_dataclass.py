from dataclasses import dataclass
from typing import Any, Callable, List, Tuple
from datetime import datetime
from aux_funcs.calculations_for_parsivel_data import volume_drop
from aux_funcs.bin_data import bin_diameter, bin_velocity
from numpy import array, cumsum, divide, fromiter, ndarray, zeros, pi
from parsivel.parsivel_dataclass import (
    ParsivelTimeStep,
    ParsivelTimeSeries,
    AREAPARSIVEL,
)
from pathlib import Path
from aux_funcs.aux_funcs_read_files import range_between_times_30s

"""
DataClass for the 3dstereo device
        "timestamp",
        "timestamp_ms",
        "diameter",
        "velocity",
        "distance_sensor",
        "shape_parameter",

I am only gonna colect the usefull ones for now.
"""

BASEAREASTEREO3D = 10000.0
MINDIST = 200.0
MAXDIST = 400.0


@dataclass(slots=True)
class Stereo3DRow:
    timestamp: int
    timestamp_ms: float
    diameter: float
    velocity: float
    distance_to_sensor: float
    shape_parameter: float


@dataclass(slots=True)
class Stereo3DSeries:
    duration: Tuple[int, int]
    series: ndarray[Stereo3DRow, Any]
    area_of_study: float

    def __getitem__(self, idx: int) -> Stereo3DRow:
        return self.series[idx]

    def __len__(self) -> int:
        return self.series.shape[0]

    def limits(self) -> Tuple[int, int]:
        if len(self) > 1:
            return (self[0].timestamp, self[-1].timestamp)
        else:
            return (0, 0)

    def limits_readable(self):
        start, finish = self.limits()
        return (datetime.utcfromtimestamp(start), datetime.utcfromtimestamp(finish))

    """
    Return the itens from each category in a numpy array
    """

    @property
    def diameters(self) -> ndarray[float, Any]:
        return fromiter((item.diameter for item in self), float)

    @property
    def velocity(self) -> ndarray[float, Any]:
        return fromiter((item.velocity for item in self), float)

    """
    Method for reading the data in its standard raw format
    """

    @classmethod
    def read_raw(cls, beggining: int, end: int, source_folder: str | Path):
        from stereo3d.read_write import stereo3d_read_from_zips

        return stereo3d_read_from_zips(beggining, end, source_folder)

    """
    Method for reading/writing the data in the pickle format
    """

    @classmethod
    def load_pickle(cls, source_folder: str | Path):
        from stereo3d.read_write import stereo_read_from_pickle

        return stereo_read_from_pickle(source_folder)

    def to_pickle(self, file_path: str | Path):
        from stereo3d.read_write import write_to_picle

        return write_to_picle(file_path, self)

    """
    Compute the rain rate in a time series
    """

    def rain_rate(self, interval_seconds: int) -> ndarray[float, Any]:
        # Define the ends of the time series
        start, stop = self.duration

        # Create an empty object with the slots to fit the data
        rain_rate = zeros(shape=((stop - start) // interval_seconds + 1,), dtype=float)

        # Loop thought every row of data and add the rate until you have a value
        for item in self:
            idx = (item.timestamp - start) // interval_seconds
            rain_rate[idx] += (
                volume_drop(item.diameter)
                / self.area_of_study
                / (interval_seconds / 3600)
            )

        return rain_rate

    def depth_for_event(self) -> float:
        return sum(volume_drop(item.diameter) / BASEAREASTEREO3D for item in self)

    def cumulative_rain_depht(self, interval_seconds: int) -> ndarray[float, Any]:
        return cumsum(self.rain_rate(interval_seconds) * interval_seconds)

    """
    Divides the distance from the sensor into ranges and counts the rain detph for 
    each range
    """

    def acumulate_by_distance(self, N: int = 1024) -> List[ndarray[float, Any]]:
        length = (MAXDIST - MINDIST) / N
        volume = zeros(shape=(N,), dtype=float)
        mean_diameter = zeros(shape=(N,), dtype=float)
        mean_velocity = zeros(shape=(N,), dtype=float)
        number_drops = zeros(shape=(N,), dtype=float)

        # Calculate the area of each session
        areas = []
        lenght = 200.0 / 1024
        for n in range(N):
            d0 = 200.0 + n * lenght
            areas.append(pi * ((d0 + lenght) ** 2 - d0**2) * 0.26525823848649227)

        for row in self:
            idx = int((row.distance_to_sensor - MINDIST) // length) - 1
            mean_diameter[idx] += row.diameter
            mean_velocity[idx] += row.velocity
            number_drops[idx] += 1

            volume[idx] += volume_drop(row.diameter)

        acumulated_depth = divide(volume, areas, where=(areas != 0))
        mean_diameter = divide(mean_diameter, number_drops, where=(number_drops != 0))
        mean_velocity = divide(mean_velocity, number_drops, where=(number_drops != 0))

        return [volume, acumulated_depth, number_drops, mean_diameter, mean_velocity]

    """
    Converts the data from the 3D stereo to the parsivel format arranging the 
    drops in the standard matrix.
    """

    def convert_to_parsivel(self) -> ParsivelTimeSeries:
        from stereo3d.convert_to_parsivel import convert_to_parsivel

        return convert_to_parsivel(self)

    """
    New additions
    """

    def filter_by_distance_to_sensor(self, new_limits: Tuple[float, float]):
        left, right = new_limits
        assert (
            new_limits[0] < new_limits[1]
        ), " The left bound has to be bigger than the right!"
        assert 200.0 <= new_limits[0], "The left bound can't be smaller than 200"
        assert new_limits[1] <= 400.0, "The right bound can't be bigger than 400"

        new_area = (
            BASEAREASTEREO3D * (right**2 - left**2) / (MAXDIST**2 - MINDIST**2)
        )

        return Stereo3DSeries(
            self.duration,
            array([item for item in self if left <= item.distance_to_sensor <= right]),
            new_area,
        )
