from pathlib import Path
from sprint06.divide_events import (
    classify_events_by_rain_rate,
    group_events_by_rain_rate,
)

from parsivel import pars_read_from_pickle
from stereo3d import stereo_read_from_pickle

parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/parsivel/"
)
stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/sprint05/stereo/"
)

parsivel_events = [
    pars_read_from_pickle(file) for file in parsivel_events_folder.iterdir()
]
stereo_events = [
    stereo_read_from_pickle(file) for file in stereo_events_folder.iterdir()
]


classification = classify_events_by_rain_rate(parsivel_events)


parsivel_groupping = group_events_by_rain_rate(parsivel_events, classification)
stereo_groupping = group_events_by_rain_rate(parsivel_events, classification)

for key in parsivel_groupping.keys():
    print(
        f"n events {key}: {len(parsivel_groupping[key])} == {len(stereo_groupping[key])}"
    )
