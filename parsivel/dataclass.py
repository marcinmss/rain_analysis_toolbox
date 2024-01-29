from dataclasses import dataclass
from typing import Any, List, NamedTuple, Tuple
from pathlib import Path
from numpy import (
    array,
    cumsum,
    mean,
    nan,
    ndarray,
    zeros,
    sum as npsum,
    abs as npabs,
)
from aux_funcs.aux_datetime import tstamp_to_readable
from matplotlib.axes import Axes


"""
The dataclass for extracting information from a parsivle file
"""
BASEPARSIVELSTYLE = {"color": "orangered"}
PARSIVELBASEAREA = 5400


class ParsivelTimeStep(NamedTuple):
    timestamp: int  # %Y%m%d%H%M%S ex: 20220402000030
    rain_rate: float  # mm
    temperature: float  # ºC
    matrix: ndarray[float, Any]  # 32x32 matrix

    @property
    def ndrops(self) -> float:
        return npsum(npabs(self.matrix), dtype=float)

    @property
    def volume_mm3(self) -> float:
        from parsivel.indicators import matrix_to_volume

        return matrix_to_volume(self.matrix)

    @classmethod
    def empty(cls, timestamp: int):
        return ParsivelTimeStep(timestamp, nan, nan, zeros((32, 32), dtype=int))

    @classmethod
    def zero_like(cls, timestamp: int):
        return ParsivelTimeStep(timestamp, 0.0, 0.0, zeros((32, 32), dtype=int))


@dataclass(slots=True)
class ParsivelTimeSeries:
    device: str
    duration: Tuple[int, int]  # beggining and end of the time series
    missing_time_steps: list  # the missing steps represented in timestamp
    series: list
    resolution_seconds: int
    area_of_study: float

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, index: int) -> ParsivelTimeStep:
        return self.series[index]

    def __str__(self) -> str:
        return (
            "Parsivel Time series:\n"
            f"  span: {self.duration_readable}, \n"
            f"  # missing_time_steps: {len(self.missing_time_steps)}, \n"
        )

    """
    Method for reading/writing the data in the pickle format
    """

    def to_pickle(self, file_path: str | Path):
        from parsivel.read_write import write_to_picle

        return write_to_picle(file_path, self)

    """
    Rain indicators
    """

    def rain_rate(self) -> ndarray[float, Any]:
        from parsivel.indicators import matrix_to_volume

        volume_series = array([matrix_to_volume(tstep.matrix) for tstep in self])
        return volume_series / self.area_of_study / self.resolution_seconds * 3600

    def avg_rain_rate(self) -> float:
        return mean(self.rain_rate(), dtype=float)

    def cumulative_depth(self) -> ndarray[float, Any]:
        from parsivel.indicators import matrix_to_volume

        volume_series = array(
            [matrix_to_volume(matrix) for matrix in self.get_matrices_series()]
        )
        return cumsum(volume_series / self.area_of_study)

    def get_nd(self) -> Tuple[ndarray, ndarray]:
        from parsivel.indicators import get_nd

        return get_nd(self)

    def get_nd3(self) -> Tuple[ndarray, ndarray]:
        from parsivel.indicators import get_nd3

        return get_nd3(self)

    def total_depth_for_event(self) -> float:
        from parsivel.indicators import matrix_to_volume

        return matrix_to_volume(self.matrix_for_event) / self.area_of_study

    def mean_diameter_for_event(self) -> float:
        from parsivel.indicators import get_mean_diameter

        return get_mean_diameter(self)

    def mean_velocity_for_event(self) -> float:
        from parsivel.indicators import get_mean_velocity

        return get_mean_velocity(self)

    """
    Other
    """

    def rain_rate_from_file(self) -> ndarray[float, Any]:
        return array([item.rain_rate for item in self])

    def percentage_zeros(self) -> float:
        n_zeros = sum(1 for is_rain in self.rain_rate() if is_rain == 0)
        return n_zeros / len(self)

    def kinetic_energy_flow_for_event(self) -> float:
        from parsivel.indicators import get_kinetic_energy

        return get_kinetic_energy(self) / (self.area_of_study * 1e-6)

    def calculated_rain_depth(self) -> ndarray[float, Any]:
        return cumsum(array(self.rain_rate() * self.resolution_seconds / 3600))

    def temperature(self) -> ndarray[ndarray, Any]:
        return array([item.temperature for item in self])

    def get_matrices_series(self) -> ndarray[ndarray, Any]:
        return array([item.matrix for item in self])

    def ndrops_in_each_class(self):
        from parsivel.indicators import get_ndrops_in_each_diameter

        return get_ndrops_in_each_diameter(self)

    """
    Methods for providing/calculating basic information about the series
    """

    @property
    def time_elapsed_seconds(self) -> List[int]:
        length = len(self.series) + len(self.missing_time_steps)
        return [i * 30 for i in range(length)]

    @property
    def matrix_for_event(self) -> ndarray:
        matrix = sum((item.matrix for item in self))
        assert isinstance(matrix, ndarray)
        return matrix

    @property
    def duration_readable(self) -> str:
        start = tstamp_to_readable(self.duration[0])
        finish = tstamp_to_readable(self.duration[1])
        return f"{start} to {finish}"

    """
    Extract events of the series
    """

    def find_events(
        self,
        dry_period_min: int = 15,
        threshold: float = 0.7,
        tinterval_sec: float = 30,
    ):
        from aux_funcs.extract_events import is_event

        return is_event(self.rain_rate(), dry_period_min, threshold, tinterval_sec)

    def extract_events_from_events_series(self, event_series: ndarray):
        from parsivel.extract_events import extract_events_from_events_series

        return extract_events_from_events_series(self, event_series)

    """
    Implementing filters to the series
    """

    def filter_by_parsivel_resolution(self):
        from parsivel.filters import resolution_filter

        return resolution_filter(self)

    """
    Methods for plotting the parsivel data
    """

    def plot_rain_rate(self, ax: Axes, style: dict = BASEPARSIVELSTYLE):
        from parsivel.plots import plot_rain_rate

        plot_rain_rate(self, ax, style)
