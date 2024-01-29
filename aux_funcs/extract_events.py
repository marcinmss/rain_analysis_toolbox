from typing import List
from numpy import array, ndarray, full


"""
Takes the rain-rate time series and returns a bolean array of the same 
shape that is true if the time step is in an event and false if it is not.
"""


def is_event(
    rain_rate: ndarray,
    dry_period_min: int = 15,
    threshold: float = 0.7,
    resolution_s: float = 30,
):
    tinterval = resolution_s / 3600
    buffer = int(dry_period_min * 60 / resolution_s)
    rains = rain_rate > 0.001
    rains_inside_buffer = array(
        [any(rains[i - buffer : i + buffer]) for i in range(rains.size)]
    )

    output = full(rain_rate.shape, False)
    first, last = 0, 0
    event_depht = 0
    in_event = False
    for i, (r, rib) in enumerate(zip(rains, rains_inside_buffer)):
        if not in_event:
            if r:
                first = i
                in_event = True
        else:
            if r:
                last = i
                event_depht += rain_rate[i] * tinterval
            elif not rib:
                if event_depht >= threshold:
                    output[first:last] = [True for _ in range(first, last)]
                in_event = False
                event_depht = 0
                first, last = 0, 0

    if first != 0 and last == 0:
        if event_depht >= threshold:
            output[first:] = True
        pass

    return output


def separate_events(is_event: ndarray) -> List[ndarray]:
    events = []
    event_idxs = []
    previous = False
    for i, tstep_is_event in enumerate(is_event):
        if tstep_is_event:
            event_idxs.append(i)
        elif previous:
            events.append(array(event_idxs))
            event_idxs = []

    if len(event_idxs) > 0:
        events.append(array(event_idxs))

    return events
