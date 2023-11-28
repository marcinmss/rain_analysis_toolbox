from dataclasses import dataclass
from typing import Any, List, Literal, NamedTuple, Tuple
from pathlib import Path
from numpy import array, cumsum, empty, nan, ndarray, zeros, sum as npsum, abs as npabs
from aux_funcs.aux_datetime import tstamp_to_readable

"""
The dataclass for extracting information from a parsivle file
"""
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
        from aux_funcs.calculations_for_parsivel_data import matrix_to_volume

        return matrix_to_volume(self.matrix)

    @classmethod
    def empty(cls, timestamp: int):
        return ParsivelTimeStep(timestamp, nan, nan, zeros((32, 32), dtype=int))

    @classmethod
    def zero_like(cls, timestamp: int):
        return ParsivelTimeStep(timestamp, 0.0, 0.0, zeros((32, 32), dtype=int))


@dataclass
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
    Method for reading the data in its standard raw format
    """

    @classmethod
    def read_raw(cls, beggining: int, end: int, source_folder: str | Path):
        from parsivel.read_write import pars_read_from_zips

        return pars_read_from_zips(beggining, end, source_folder)

    """
    Method for reading/writing the data in the pickle format
    """

    @classmethod
    def load_pickle(cls, source_folder: str | Path):
        from parsivel.read_write import pars_read_from_pickle

        return pars_read_from_pickle(source_folder)

    def to_pickle(self, file_path: str | Path):
        from parsivel.read_write import write_to_picle

        return write_to_picle(file_path, self)

    """
    Methods for providing/calculating basic information about the series
    """

    @property
    def rain_rate(self) -> ndarray[float, Any]:
        from aux_funcs.calculations_for_parsivel_data import matrix_to_volume

        volume_series = array([matrix_to_volume(matrix) for matrix in self.matrices])
        return volume_series / self.area_of_study / self.resolution_seconds * 3600

    @property
    def total_depth_for_event(self) -> float:
        from aux_funcs.calculations_for_parsivel_data import matrix_to_volume

        return matrix_to_volume(self.matrix_for_event) / self.area_of_study

    @property
    def npa(self) -> ndarray[float, Any]:
        area_m2 = self.area_of_study * 1e-6
        return array([item.ndrops for item in self]) / area_m2

    @property
    def npa_for_event(self) -> float:
        area_m2 = self.area_of_study * 1e-6
        return npsum(self.matrix_for_event) / area_m2

    @property
    def files_rain_rate(self) -> ndarray[float, Any]:
        return array([item.rain_rate for item in self])

    @property
    def mean_diameter_for_event(self) -> float:
        from parsivel.indicators import get_mean_diameter

        return get_mean_diameter(self)

    @property
    def mean_velocity_for_event(self) -> float:
        from parsivel.indicators import get_mean_velocity

        return get_mean_velocity(self)

    @property
    def cumulative_rain_depth(self) -> ndarray[float, Any]:
        from aux_funcs.calculations_for_parsivel_data import matrix_to_volume

        volume_series = array([matrix_to_volume(matrix) for matrix in self.matrices])
        return cumsum(volume_series / self.area_of_study)

    @property
    def kinetic_energy_flow_for_event(self) -> float:
        from parsivel.indicators import get_kinetic_energy

        return get_kinetic_energy(self) / (self.area_of_study * 1e-6)

    @property
    def calculated_rain_depth(self) -> ndarray[float, Any]:
        return cumsum(array(self.rain_rate * self.resolution_seconds / 3600))

    @property
    def temperature(self) -> ndarray[ndarray, Any]:
        return array([item.temperature for item in self])

    @property
    def matrices(self) -> ndarray[ndarray, Any]:
        return array([item.matrix for item in self])

    @property
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
    def matrix_for_event(self) -> ndarray | Literal[0]:
        return sum((item.matrix for item in self))

    @property
    def duration_readable(self) -> str:
        start = tstamp_to_readable(self.duration[0])
        finish = tstamp_to_readable(self.duration[1])
        return f"{start} to {finish}"

    """
    Extract events of the series
    """

    def exstract_events(
        self, minimal_time_lenght_min: int, buffer_min: int, threshold: float
    ):
        from parsivel.extract_events import extract_events

        return extract_events(self, minimal_time_lenght_min, buffer_min, threshold)

    def find_events(
        self,
        dry_period_min: int = 15,
        threshold: float = 0.7,
        tinterval_sec: float = 30,
    ):
        from aux_funcs.extract_events import is_event

        return is_event(self.rain_rate, dry_period_min, threshold, tinterval_sec)

    def colect_events(self, event_series: ndarray):
        from parsivel.extract_events import grab_events

        return grab_events(self, event_series)

    """
    Implementing filters to the series
    """

    def filter_by_parsivel_resolution(self):
        from parsivel.filters import resolution_filter

        return resolution_filter(self)
