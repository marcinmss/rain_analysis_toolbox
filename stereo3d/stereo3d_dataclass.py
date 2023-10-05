from dataclasses import dataclass
from typing import Any, List, Tuple
from datetime import datetime
from aux_funcs.calculations_for_parsivel_data import volume_drop
from aux_funcs.bin_data import bin_diameter, bin_velocity
from numpy import array, cumsum, divide, fromiter, ndarray, zeros
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
    Compute the rain rate in a time series
    """

    def rain_rate(self, interval_seconds: int) -> ndarray[float, Any]:
        # Define the ends of the time series
        start, stop = self.duration

        # Create an empty object with the slots to fit the data
        rain_rate = zeros(shape=((stop - start) // interval_seconds,), dtype=float)

        # Loop thought every row of data and add the rate until you have a value
        for item in self:
            idx = (item.timestamp - start) // interval_seconds
            rain_rate[idx] += (
                volume_drop(item.diameter) / AREA3DSTEREO / (interval_seconds / 3600)
            )

        return rain_rate

    def cumulative_rain_depht(self, interval_seconds: int) -> ndarray[float, Any]:
        return cumsum(self.rain_rate(interval_seconds))

    """
    Divides the distance from the sensor into ranges and counts the rain detph for 
    each range
    """

    def acumulate_by_distance(self) -> List[ndarray[float, Any]]:
        N = 1024
        depth = zeros(shape=(N,), dtype=float)
        length = (MAXDIST - MINDIST) / N
        mean_diameter = zeros(shape=(N,), dtype=float)
        number_drops = zeros(shape=(N,), dtype=float)

        for row in self:
            idx = int((row.distance_to_sensor - MINDIST) // length) - 1
            mean_diameter[idx] += row.diameter
            number_drops[idx] += 1

            depth[idx] += volume_drop(row.diameter) / AREA3DSTEREO

        mean_diameter = divide(mean_diameter, number_drops, where=(number_drops != 0))

        return [depth, number_drops, mean_diameter]

    """
    Converts the data from the 3D stereo to the parsivel format arranging the 
    drops in the standard matrix.
    """

    def convert_to_parsivel(self) -> ParsivelTimeSeries:
        start = datetime.utcfromtimestamp(self.duration[0])
        finish = datetime.utcfromtimestamp(self.duration[1])
        range_timesteps = (
            item.timestamp() for item in range_between_times_30s(start, finish)
        )
        series: ndarray[ParsivelInfo, Any] = array(
            [ParsivelInfo.zero_like(int(timestamp)) for timestamp in range_timesteps],
            dtype=ParsivelInfo,
        )

        factor = AREAPARSIVEL / AREA3DSTEREO
        # factor = 1
        for item in self:
            # generate the matrix for the
            idx = int((item.timestamp - self.duration[0]) // 30)
            class_velocity = bin_velocity(item.velocity)
            class_diameter = bin_diameter(item.diameter)

            if 1 <= class_velocity <= 32 and 0 < class_diameter < 33:
                series[idx].matrix[class_diameter - 1, class_velocity - 1] += factor

        return ParsivelTimeSeries("3D Stereo", self.duration, [], series.tolist(), 30)
