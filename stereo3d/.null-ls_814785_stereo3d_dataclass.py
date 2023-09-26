from dataclasses import dataclass
from typing import Any, Tuple
from datetime import datetime
from aux_funcs.calculations_for_parsivel_data import volume_drop
from aux_funcs.bin_data import bin_diameter, bin_velocity
from numpy import array, cumsum, fromiter, ndarray, zeros
from parsivel.parsivel_dataclass import ParsivelInfo, ParsivelTimeSeries, AREAPARSIVEL
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
AREA3DSTEREO = 10000.0


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

    def diameter(self) -> ndarray[float, Any]:
        return fromiter((item.diameter for item in self), float)

    def velocity(self) -> ndarray[float, Any]:
        return fromiter((item.velocity for item in self), float)

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
                volume_drop(item.diameter) / AREA3DSTEREO / (interval_seconds / 3600)
            )

        return rain_rate

    def cumulative_rain_depht(self, interval_seconds: int) -> ndarray[float, Any]:
        return cumsum(self.rain_rate(interval_seconds))

    # TODO: get more data other then the matrix
    def convert_to_parsivel(self) -> ParsivelTimeSeries:
        start = datetime.utcfromtimestamp(self.duration[0])
        finish = datetime.utcfromtimestamp(self.duration[1])
        range_timesteps = [
            item.timestamp() for item in range_between_times_30s(start, finish)
        ]
        series: ndarray[ParsivelInfo, Any] = array(
            [ParsivelInfo.empty(int(timestamp)) for timestamp in range_timesteps],
            dtype=ParsivelInfo,
        )

        factor = AREAPARSIVEL / AREA3DSTEREO
        for item in self:
            idx = int((item.timestamp - self.duration[0]) // 30)
            class_velocity = bin_velocity(item.velocity)
            class_diameter = bin_diameter(item.diameter)

            if 1 <= class_velocity <= 32 and 0 < class_diameter < 33:
                series[idx].matrix[class_diameter - 1, class_velocity - 1] += factor

        return ParsivelTimeSeries(self.duration, [], series.tolist())
