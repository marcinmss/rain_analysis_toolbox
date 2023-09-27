from dataclasses import dataclass
from typing import Any, List, Literal, Tuple
from numpy import array, cumsum, empty, nan, ndarray, zeros
from aux_funcs.calculations_for_parsivel_data import (
    AREAPARSIVEL,
    matrix_to_rainrate,
    matrix_to_rainrate2,
)

"""
The dataclass for extracting information from a parsivle file
"""


@dataclass
class ParsivelInfo:
    timestamp: int  # %Y%m%d%H%M%S ex: 20220402000030
    rain_rate: float  # mm
    temperature: float  # ºC
    matrix: ndarray  # 32x32 matrix

    def calculated_rate(self):
        return matrix_to_rainrate(self.matrix, AREAPARSIVEL)

    def calculated_rate2(self):
        return matrix_to_rainrate2(self.matrix, AREAPARSIVEL)

    @classmethod
    def empty(cls, timestamp: int):
        return ParsivelInfo(timestamp, nan, nan, empty((32, 32), dtype=float))

    @classmethod
    def zero_like(cls, timestamp: int):
        return ParsivelInfo(timestamp, 0.0, 0.0, zeros((32, 32), dtype=float))


@dataclass
class ParsivelTimeSeries:
    duration: Tuple[int, int]  # beggining and end of the time series
    missing_time_steps: list  # the missing steps represented in timestamp
    series: list
    resolution_seconds: int

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, index: int) -> ParsivelInfo:
        return self.series[index]

    """
    Methods for providing/calculating basic information about the series
    """

    def rain_rate(self) -> ndarray[float, Any]:
        return array([item.rain_rate for item in self])

    def calculated_rate(self) -> ndarray[float, Any]:
        return array([item.calculated_rate() for item in self])

    def get_calculated_rate2(self) -> ndarray[float, Any]:
        return array([item.calculated_rate2() for item in self])

    def get_cumulative_rain_depth(self) -> ndarray[float, Any]:
        return cumsum([item.rain_rate * 30.0 for item in self])

    def calculated_rain_depth(self) -> ndarray[float, Any]:
        return cumsum([item.calculated_rate() * 30.0 for item in self])

    def calculated_rain_depth2(self) -> ndarray[float, Any]:
        return cumsum([item.calculated_rate2() * 30.0 for item in self])

    def get_sensor_temperature(self) -> ndarray[ndarray, Any]:
        return array([item.temperature for item in self])

    def get_sdd_matrix(self) -> ndarray[ndarray, Any]:
        return array([item.matrix for item in self])

    def get_time_elapsed_seconds(self) -> List[int]:
        length = len(self.series) + len(self.missing_time_steps)
        return [i * 30 for i in range(length)]

    def get_overall_matrix(self) -> ndarray | Literal[0]:
        return sum((item.matrix for item in self))

    """
    Change the resolution of the series
    """

    def change_resolution(self, new_resolution_seconds: int) -> ParsivelTimeSeries:
        assert new_resolution_seconds % self.resolution_seconds == 0
        factor = new_resolution_seconds // self.resolution_seconds
        return self
