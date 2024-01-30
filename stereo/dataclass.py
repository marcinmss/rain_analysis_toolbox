from dataclasses import dataclass
from typing import Any, Tuple
from matplotlib.axes import Axes
from aux_funcs.general import volume_drop
from numpy import array, cumsum, fromiter, mean, ndarray
from parsivel.dataclass import ParsivelTimeSeries
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

BASESTEREOSTYLE = {"color": "dodgerblue"}
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
        from stereo.distance_analisys import area_of_session

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
    Method for reading/writing the data in the pickle format
    """

    def to_pickle(self, file_path: str | Path):
        from stereo.read_write import write_to_picle

        return write_to_picle(file_path, self)

    """
    Methods for calculating indicators from the data
    """

    def rain_rate(self, interval_seconds: float = 30) -> ndarray[float, Any]:
        from stereo.indicators import rain_rate

        return rain_rate(self, interval_seconds)

    def time_series(self, interval_seconds: float = 30) -> ndarray[float, Any]:
        from stereo.indicators import time_series

        return time_series(self, interval_seconds)

    def mean_diameter_for_event(self):
        return mean(self.diameters)

    def mean_velocity_for_event(self):
        return mean(self.velocity)

    def kinetic_energy_flow(self) -> float:
        from stereo.indicators import get_kinetic_energy

        return get_kinetic_energy(self) / (self.area_of_study * 1e-6)

    def total_depth_for_event(self) -> float:
        return sum(volume_drop(item.diameter) / self.area_of_study for item in self)

    def cumulative_rain_depht(self, interval_seconds: int = 30) -> ndarray[float, Any]:
        return cumsum(self.rain_rate(interval_seconds) * interval_seconds / 3600)

    @property
    def ndrops_in_each_diameter_class(self):
        from stereo.indicators import get_ndrops_in_diameter_classes

        return get_ndrops_in_diameter_classes(self)

    @property
    def avg_rain_rate(self) -> float:
        return mean(self.rain_rate(), dtype=float)

    @property
    def percentage_zeros(self) -> float:
        n_zeros = sum(1 for is_rain in self.rain_rate() if is_rain == 0)
        return n_zeros / len(self)

    """
    Plot basic plot for indicators
    """

    def plot_rain_rate(self, ax: Axes, style: dict = BASESTEREOSTYLE):
        from stereo.plots import plot_rain_rate

        plot_rain_rate(self, ax, style)

    """
    Divides the distance from the sensor into ranges and counts the rain detph for 
    each range
    """

    def split_by_distance_to_sensor(self, number_of_splits: int = 8):
        from stereo.distance_analisys import split_by_distance_to_sensor

        return split_by_distance_to_sensor(self, number_of_splits)

    """
    Converts the data from the 3D stereo to the parsivel format arranging the 
    drops in the standard matrix.
    """

    def convert_to_parsivel(self) -> ParsivelTimeSeries:
        from stereo.convert_to_parsivel import convert_to_parsivel

        return convert_to_parsivel(self)

    """
    Creates a new series with only with the drops within a certein distance
    """

    def filter_by_distance_to_sensor(self, new_limits: Tuple[float, float]):
        from stereo.distance_analisys import filter_by_distance_to_sensor

        return filter_by_distance_to_sensor(self, new_limits)

    """
    Creates a new series with only with the drops within a certein distance
    """

    def filter_to_parsivel_resolution(self):
        from stereo.filters import parsivel_filter

        return parsivel_filter(self)

    """
    New additions
    """

    def extract_events(self, is_event_array: ndarray):
        from stereo.events import extract_events_from_is_event_series

        return extract_events_from_is_event_series(is_event_array, self)

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
