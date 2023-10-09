from typing import List
from parsivel2.parsivel_dataclass2 import ParsivelTimeSeries, ParsivelTimeStep


# TODO: Deal with the missing timesteps
def extract_events(
    series: ParsivelTimeSeries,
    minimal_length_minutes: int,
    buffer_minutes: int,
    threshold: float,
) -> List[ParsivelTimeSeries]:
    factor = (buffer_minutes * 60) // series.resolution_seconds

    is_it_raining = [tstep.rain_rate > threshold for tstep in series]

    is_event = [any(is_it_raining[i - factor : i + factor]) for i in range(len(series))]

    events = []
    event: List[ParsivelTimeStep] = []
    previous = False
    for tstep, is_rainin in zip(series.series, is_event):
        if is_rainin is True:
            event.append(tstep)
        elif is_rainin is False and previous is True:
            # Add the event to the output
            events.append(
                ParsivelTimeSeries(
                    series.device,
                    (event[0].timestamp, event[-1].timestamp),
                    [],
                    event,
                    series.resolution_seconds,
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
            )
        )

    # Minimal length
    min_factor = (
        (2 * buffer_minutes + minimal_length_minutes) * 60 // series.resolution_seconds
    )
    events = [item for item in events if len(item) > min_factor]
    return events
