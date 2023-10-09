from dataclasses import dataclass
from typing import Any, List, Literal, Tuple
from numpy import array, cumsum, empty, fromiter, nan, ndarray, zeros
from aux_funcs.calculations_for_parsivel_data import (
    AREAPARSIVEL,
    matrix_to_rainrate,
    matrix_to_rainrate2,
)
from aux_funcs.aux_datetime import standard_to_rounded_tstamp, tstamp_to_readable
from itertools import groupby

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

    @property
    def rain_rate(self) -> ndarray[float, Any]:
        return array([item.rain_rate for item in self])

    @property
    def calculated_rate(self) -> ndarray[float, Any]:
        return array([item.calculated_rate() for item in self])

    @property
    def calculated_rate2(self) -> ndarray[float, Any]:
        return array([item.calculated_rate2() for item in self])

    @property
    def cumulative_rain_depth(self) -> ndarray[float, Any]:
        return cumsum([item.rain_rate * self.resolution_seconds for item in self])

    @property
    def calculated_rain_depth(self) -> ndarray[float, Any]:
        return cumsum(
            [item.calculated_rate() * self.resolution_seconds for item in self]
        )

    @property
    def calculated_rain_depth2(self) -> ndarray[float, Any]:
        return cumsum(
            [item.calculated_rate2() * self.resolution_seconds for item in self]
        )

    @property
    def get_sensor_temperature(self) -> ndarray[ndarray, Any]:
        return array([item.temperature for item in self])

    @property
    def get_sdd_matrix(self) -> ndarray[ndarray, Any]:
        return array([item.matrix for item in self])

    @property
    def get_time_elapsed_seconds(self) -> List[int]:
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
    Shrink the series
    """

    def change_duration(self, new_start: int, new_finish: int):
        # Turns the inputs into rounded timestamps
        ns_tstamp = standard_to_rounded_tstamp(new_start, self.resolution_seconds)
        nf_tstamp = standard_to_rounded_tstamp(new_finish, self.resolution_seconds)

        # Check if the given dates are in the range
        assert (
            new_start < self.duration[0] or self.duration[1] < new_finish
        ), "To change duration the old interval needs to contain the new one."

        # Filter for the points that are in the given range
        new_series = [item for item in self if ns_tstamp < item.timestamp < nf_tstamp]

        # Filter the missing timesteps that are in the range
        new_missing_time_steps = [
            tstep for tstep in self.missing_time_steps if ns_tstamp < tstep < nf_tstamp
        ]

        return ParsivelTimeSeries(
            (ns_tstamp, nf_tstamp),
            new_missing_time_steps,
            new_series,
            self.resolution_seconds,
        )

    """
    Extract events of the series
    """

    # def exstract_events(self, threshold: float, buffer: int):
    #     # TODO: checks if the buffer is a multiple of the resolution
    #     n = buffer // self.resolution_seconds
    #     is_rainning = self.rain_rate >= threshold
    #     is_event = [any(is_rainning[i - n : i + n] for i in range(len(is_rainning)))]

    """
    Change the resolution of the series
    """

    def change_resolution(self, new_resolution_seconds: int):
        if new_resolution_seconds % self.resolution_seconds != 0:
            error_msg = (
                f" Factor new resolution ({new_resolution_seconds}) must"
                "be a multiple of the old one ({self.resolution_seconds})"
            )
            assert False, error_msg

        factor = new_resolution_seconds // self.resolution_seconds

        new_series = [
            agregate_data(self.series[i : i + factor])
            for i in range(0, len(self), factor)
        ]

        new_missing_time_steps: List[int] = [
            ((time_step - self.duration[0]) // new_resolution_seconds)
            * new_resolution_seconds
            + self.duration[0]
            for time_step in self.missing_time_steps
        ]

        return ParsivelTimeSeries(
            self.duration, new_missing_time_steps, new_series, new_resolution_seconds
        )


def agregate_data(data: List[ParsivelInfo]) -> ParsivelInfo:
    n = len(data)
    timestamp = data[0].timestamp
    temp = sum((item.temperature for item in data)) / n
    matrix = sum((item.matrix for item in data))
    assert isinstance(matrix, ndarray)
    rate = sum((item.rain_rate for item in data)) / n
    return ParsivelInfo(timestamp, rate, temp, matrix)
