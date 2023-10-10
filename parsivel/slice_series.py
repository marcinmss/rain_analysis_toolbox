"""
Shrink the series
"""

# def change_duration(self, new_start: int, new_finish: int):
#     # Turns the inputs into rounded timestamps
#     ns_tstamp = standard_to_rounded_tstamp(new_start, self.resolution_seconds)
#     nf_tstamp = standard_to_rounded_tstamp(new_finish, self.resolution_seconds)
#
#     # Check if the given dates are in the range
#     assert (
#         new_start < self.duration[0] or self.duration[1] < new_finish
#     ), "To change duration the old interval needs to contain the new one."
#
#     # Filter for the points that are in the given range
#     new_series = [item for item in self if ns_tstamp < item.timestamp < nf_tstamp]
#
#     # Filter the missing timesteps that are in the range
#     new_missing_time_steps = [
#         tstep for tstep in self.missing_time_steps if ns_tstamp < tstep < nf_tstamp
#     ]
#
#     return ParsivelTimeSeries(
#         self.device,
#         (ns_tstamp, nf_tstamp),
#         new_missing_time_steps,
#         new_series,
#         self.resolution_seconds,
#     )

"""
Change the resolution of the series
"""

# def change_resolution(self, new_resolution_seconds: int):
#     if new_resolution_seconds % self.resolution_seconds != 0:
#         error_msg = (
#             f" Factor new resolution ({new_resolution_seconds}) must"
#             "be a multiple of the old one ({self.resolution_seconds})"
#         )
#         assert False, error_msg
#
#     factor = new_resolution_seconds // self.resolution_seconds
#
#     new_series = [
#         agregate_data(self.series[i : i + factor])
#         for i in range(0, len(self), factor)
#     ]
#
#     new_missing_time_steps: List[int] = [
#         ((time_step - self.duration[0]) // new_resolution_seconds)
#         * new_resolution_seconds
#         + self.duration[0]
#         for time_step in self.missing_time_steps
#     ]
#
#     return ParsivelTimeSeries(
#         self.device,
#         self.duration,
#         new_missing_time_steps,
#         new_series,
#         new_resolution_seconds,
#     )
#


# def agregate_data(data: List[ParsivelInfo]) -> ParsivelInfo:
#     n = len(data)
#     timestamp = data[0].timestamp
#     temp = sum((item.temperature for item in data)) / n
#     matrix = sum((item.matrix for item in data))
#     assert isinstance(matrix, ndarray)
#     rate = sum((item.rain_rate for item in data)) / n
#     return ParsivelInfo(timestamp, rate, temp, matrix)
# def agregate_data(data: List[ParsivelTimeStep]) -> ParsivelTimeStep:
#     n = len(data)
#     timestamp = data[0].timestamp
#     temp = 0.0
#     matrix = zeros((32, 32))
#     rate = 0.0
#     for item in data:
#         temp += item.temperature / n
#         matrix += item.matrix
#         rate += item.rain_rate
#
#     return ParsivelTimeStep(timestamp, rate, temp, matrix)
