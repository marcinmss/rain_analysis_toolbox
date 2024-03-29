from typing import List, Tuple
from stereo import Stereo3DSeries, BASEAREASTEREO3D
from numpy import array, ndarray, zeros
from aux_funcs.general import volume_drop


"""
Takes the a large series and the is_event array and divide the large series into smaller ones based on the events detected.
"""


def extract_events_from_is_event_series(
    is_event_series: ndarray,
    series: Stereo3DSeries,
):
    # Take the time stamps from the beggining and end of every
    last_one = True
    beg = 0
    list_durations = []
    time_stamps = series.time_series()
    for curr_one, curr_tstep in zip(is_event_series, time_stamps):
        if last_one is False and curr_one is True:
            beg = curr_tstep
        elif last_one is True and curr_one is False:
            list_durations.append((beg, curr_tstep - 30))
        else:
            last_one = curr_one

    return extract_events_from_duration(series, list_durations)


"""
Use the function to extract events given the intervals of beggining and end.
"""


def extract_events_from_duration(
    series: Stereo3DSeries, events_duration: List[Tuple[int, int]]
) -> List[Stereo3DSeries]:
    output = []
    for beg, end in events_duration:
        output.append(
            Stereo3DSeries(
                series.device,
                (beg, end),
                array([item for item in series if beg <= item.timestamp < end]),
                series.limits_area_of_study,
            )
        )
    return output


"""
Use this function to detect events utilizing the variables given bellow
"""


def find_events(
    series: Stereo3DSeries,
    minimal_length_minutes: int,
    buffer_minutes: int,
    minimal_depth_per_event: float,
    threshold: float,
) -> List[Stereo3DSeries]:
    interval_seconds = 30
    factor = (buffer_minutes * 60) // interval_seconds

    # Define the ends of the time series
    start, stop = series.duration[0], series.duration[1]

    # Create an empty object with the slots to fit the data
    rain_rate = zeros(shape=((stop - start) // interval_seconds,), dtype=float)

    # Loop thought every row of data and add the rate until you have a value
    for item in series:
        idx = (item.timestamp - start) // interval_seconds
        rain_rate[idx] += (
            volume_drop(item.diameter) / BASEAREASTEREO3D / (interval_seconds / 3600)
        )

    is_it_raining = [rate > threshold for rate in rain_rate]

    is_in_event = [
        any(
            is_it_raining[
                (row.timestamp - start) // interval_seconds
                - factor : (row.timestamp - start) // interval_seconds
                + factor
            ]
        )
        for row in series
    ]

    duration_events: List[Tuple[int, int]] = []
    beg = 0
    previous = False
    for is_event, row in zip(is_in_event, series.series):
        if previous is False and is_event is True:
            beg = (row.timestamp - start) // interval_seconds * interval_seconds + start

        elif previous is True and is_event is False:
            end = (row.timestamp - start) // interval_seconds * interval_seconds + start

            duration_events.append((beg, end))
            beg = 0
        previous = is_event
    if beg != 0:
        duration_events.append((beg, series.duration[1]))

    output = []
    for beg, end in duration_events:
        if (end - beg) >= (minimal_length_minutes + 2 * buffer_minutes) * 60:
            new_series = Stereo3DSeries(
                series.device,
                (beg, end),
                array([item for item in series if beg <= item.timestamp < end]),
                series.limits_area_of_study,
            )
            if minimal_depth_per_event < new_series.total_depth_for_event():
                output.append(new_series.duration)

    return output
