from typing import List

from numpy import ndarray
from parsivel.dataclass import ParsivelTimeSeries, ParsivelTimeStep


def grab_events(
    series: ParsivelTimeSeries, events_series: ndarray
) -> List[ParsivelTimeSeries]:
    events = []
    event: List[ParsivelTimeStep] = []
    previous = False
    for tstep, is_rainin in zip(series.series, events_series):
        if is_rainin:
            event.append(tstep)
        elif not is_rainin and previous:
            start, finish = event[0].timestamp, event[-1].timestamp
            missing_timesteps = [
                tstamp
                for tstamp in series.missing_time_steps
                if start <= tstamp <= finish
            ]
            events.append(
                ParsivelTimeSeries(
                    series.device,
                    (start, finish),
                    missing_timesteps,
                    event,
                    series.resolution_seconds,
                    series.area_of_study,
                )
            )
            event: List[ParsivelTimeStep] = []
        previous = is_rainin

    # Adds the last period if it contains rain
    if len(event) != 0:
        events.append(
            ParsivelTimeSeries(
                series.device,
                (event[0].timestamp, event[-1].timestamp),
                [],
                event,
                series.resolution_seconds,
                series.area_of_study,
            )
        )

    return events


def extract_events(
    series: ParsivelTimeSeries,
    minimal_length_minutes: int,
    buffer_minutes: int,
    minimal_depth_mm: float,
) -> List[ParsivelTimeSeries]:
    factor = (buffer_minutes * 60) // series.resolution_seconds

    is_it_raining = [tstep.rain_rate > 0.001 for tstep in series]

    is_event = [any(is_it_raining[i - factor : i + factor]) for i in range(len(series))]

    events = []
    event: List[ParsivelTimeStep] = []
    previous = False
    for tstep, is_rainin in zip(series.series, is_event):
        if is_rainin is True:
            event.append(tstep)
        elif is_rainin is False and previous is True:
            # Add the event to the output
            start, finish = event[0].timestamp, event[-1].timestamp
            missing_timesteps = [
                tstamp
                for tstamp in series.missing_time_steps
                if start <= tstamp <= finish
            ]
            events.append(
                ParsivelTimeSeries(
                    series.device,
                    (start, finish),
                    missing_timesteps,
                    event,
                    series.resolution_seconds,
                    series.area_of_study,
                )
            )
            event: List[ParsivelTimeStep] = []
        previous = is_rainin

    # Adds the last period if it contains rain
    if len(event) != 0:
        events.append(
            ParsivelTimeSeries(
                series.device,
                (event[0].timestamp, event[-1].timestamp),
                [],
                event,
                series.resolution_seconds,
                series.area_of_study,
            )
        )

    # Minimal length
    min_factor = (
        (2 * buffer_minutes + minimal_length_minutes) * 60 // series.resolution_seconds
    )
    events = [
        item
        for item in events
        if (len(item) > min_factor and item.total_depth > minimal_depth_mm)
    ]
    return events
