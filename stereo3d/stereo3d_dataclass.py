from dataclasses import dataclass
from typing import Any, List, Tuple
from aux_funcs.calculations_for_parsivel_data import volume_drop
from numpy import array, cumsum, fromiter, mean, ndarray
from parsivel.parsivel_dataclass import ParsivelTimeSeries
from pathlib import Path

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
    device: str
    duration: Tuple[int, int]
    series: ndarray[Stereo3DRow, Any]
    limits_area_of_study: Tuple[float, float]

    def __getitem__(self, idx: int) -> Stereo3DRow:
        return self.series[idx]

    def __len__(self) -> int:
        return self.series.shape[0]

    @property
    def area_of_study(self):
        from stereo3d.distance_analisys import area_of_session

        return area_of_session(self.limits_area_of_study)

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

    def rain_rate(self, interval_seconds: int = 30) -> ndarray[float, Any]:
        from stereo3d.indicators import rain_rate

        return rain_rate(self, interval_seconds)

    @property
    def npa(self, interval_seconds: int = 30) -> ndarray[float, Any]:
        from stereo3d.indicators import get_npa

        return get_npa(self, interval_seconds)

    @property
    def npa_event(self) -> float:
        return len(self) / (self.area_of_study * 1e-6)

    @property
    def mean_diameter(self):
        return mean(self.diameters)

    @property
    def mean_velocity(self):
        return mean(self.velocity)

    @property
    def kinetic_energy_flow(self) -> float:
        from stereo3d.indicators import get_kinetic_energy

        return get_kinetic_energy(self) / (self.area_of_study * 1e-6)

    @property
    def total_rain_depth(self) -> float:
        return sum(volume_drop(item.diameter) / self.area_of_study for item in self)

    def cumulative_rain_depht(self, interval_seconds: int = 30) -> ndarray[float, Any]:
        return cumsum(self.rain_rate(interval_seconds) * interval_seconds)

    """
    Divides the distance from the sensor into ranges and counts the rain detph for 
    each range
    """

    def acumulate_by_distance(self, N: int = 1024) -> List[ndarray[float, Any]]:
        from stereo3d.distance_analisys import acumulate_by_distance

        return acumulate_by_distance(self, N)

    def split_by_distance_to_sensor(self, number_of_splits: int = 8):
        from stereo3d.distance_analisys import split_by_distance_to_sensor

        return split_by_distance_to_sensor(self, number_of_splits)

    """
    Converts the data from the 3D stereo to the parsivel format arranging the 
    drops in the standard matrix.
    """

    def convert_to_parsivel(self) -> ParsivelTimeSeries:
        from stereo3d.convert_to_parsivel import convert_to_parsivel

        return convert_to_parsivel(self)

    """
    Creates a new series with only with the drops within a certein distance
    """

    def filter_by_distance_to_sensor(self, new_limits: Tuple[float, float]):
        from stereo3d.distance_analisys import filter_by_distance_to_sensor

        return filter_by_distance_to_sensor(self, new_limits)

    """
    New additions
    """

    def extract_events(self, events_duration: List[Tuple[int, int]]):
        from stereo3d.events import extract_events

        return extract_events(self, events_duration)

    def shrink_series(self, new_limits_tstamp: Tuple[int, int]):
        new_beg, new_end = new_limits_tstamp
        old_beg, old_end = self.duration
        assert old_beg <= new_beg <= new_end <= old_end, "The limits are incorect!"

        return Stereo3DSeries(
            self.device,
            (new_beg, new_end),
            array([item for item in self if new_beg <= item.timestamp <= new_end]),
            self.limits_area_of_study,
        )
