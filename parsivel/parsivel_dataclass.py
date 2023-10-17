from dataclasses import dataclass
from typing import Any, List, Literal, NamedTuple, Tuple
from pathlib import Path
from numpy import array, cumsum, empty, nan, ndarray, zeros
from aux_funcs.calculations_for_parsivel_data import (
    AREAPARSIVEL,
    matrix_to_rainrate,
    matrix_to_rainrate2,
)
from aux_funcs.aux_datetime import tstamp_to_readable

"""
The dataclass for extracting information from a parsivle file
"""


class ParsivelTimeStep(NamedTuple):
    timestamp: int  # %Y%m%d%H%M%S ex: 20220402000030
    rain_rate: float  # mm
    temperature: float  # ºC
    matrix: ndarray[float, Any]  # 32x32 matrix

    def calculated_rate(self, area_of_study: float):
        return matrix_to_rainrate(self.matrix, area_of_study)

    def calculated_rate2(self):
        return matrix_to_rainrate2(self.matrix, AREAPARSIVEL)

    @classmethod
    def empty(cls, timestamp: int):
        return ParsivelTimeStep(timestamp, nan, nan, empty((32, 32), dtype=float))

    @classmethod
    def zero_like(cls, timestamp: int):
        return ParsivelTimeStep(timestamp, 0.0, 0.0, zeros((32, 32), dtype=float))


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
        return array([item.rain_rate for item in self])

    @property
    def calculated_rate(self) -> ndarray[float, Any]:
        return array([item.calculated_rate(self.area_of_study) for item in self])

    @property
    def cumulative_rain_depth(self) -> ndarray[float, Any]:
        return cumsum([item.rain_rate * self.resolution_seconds for item in self])

    @property
    def calculated_rain_depth(self) -> ndarray[float, Any]:
        return cumsum(
            [
                item.calculated_rate(self.area_of_study) * self.resolution_seconds
                for item in self
            ]
        )

    @property
    def temperature(self) -> ndarray[ndarray, Any]:
        return array([item.temperature for item in self])

    @property
    def matrices(self) -> ndarray[ndarray, Any]:
        return array([item.matrix for item in self])

    @property
    def time_elapsed_seconds(self) -> List[int]:
        length = len(self.series) + len(self.missing_time_steps)
        return [i * 30 for i in range(length)]

    @property
    def get_overall_matrix(self) -> ndarray | Literal[0]:
        return sum((item.matrix for item in self))

    @property
    def duration_readable(self) -> Tuple[str, str]:
        start = tstamp_to_readable(self.duration[0])
        finish = tstamp_to_readable(self.duration[1])
        return (start, finish)

    """
    Extract events of the series
    """

    def exstract_events(
        self, minimal_time_lenght_min: int, buffer_min: int, threshold: float
    ):
        from parsivel.extract_events import extract_events

        return extract_events(self, minimal_time_lenght_min, buffer_min, threshold)
