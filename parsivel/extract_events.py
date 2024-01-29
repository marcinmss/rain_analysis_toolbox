from typing import List

from numpy import ndarray
from parsivel.dataclass import ParsivelTimeSeries, ParsivelTimeStep


def extract_events_from_events_series(
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
