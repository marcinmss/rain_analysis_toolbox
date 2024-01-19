from pathlib import Path
from typing import List

from parsivel import ParsivelTimeSeries, parsivel_read_from_pickle
from stereo import Stereo3DSeries, stereo_read_from_pickle

parsivel_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/parsivel/"
)

stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo/"
)

stereo_converted_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
)


def print_general_info(
    parsivel_events: List[ParsivelTimeSeries],
    stereo_events: List[Stereo3DSeries],
    stereo_converted_events: List[ParsivelTimeSeries],
):
    # Filter the converted events for the drops outside parsivel resolution
    stereo_converted_filtered = [
        event.filter_by_parsivel_resolution() for event in stereo_converted_events
    ]

    # Print the total depth for every event
    print("Total depth for sum of events:")
    print(
        f"Parisivel: {sum(event.total_depth_for_event() for event in parsivel_events)}"
    )
    print(f"3D Stereo: {sum(event.total_depth_for_event() for event in stereo_events)}")
    print(
        f"3D Stereo Filtered: {sum(event.total_depth_for_event() for event in stereo_converted_filtered)}"
    )


if __name__ == "__main__":
    # Read the data for all 3 types
    print("Reading Parsivel Data")
    parsivel_events = [
        parsivel_read_from_pickle(file)
        for file in sorted(parsivel_events_folder.iterdir())
    ]
    print("Done.")

    print("Reading 3D Stereo Data")
    stereo_events = [
        stereo_read_from_pickle(file) for file in sorted(stereo_events_folder.iterdir())
    ]
    print("Done.")

    print("Reading 3D Stereo converted Data")
    stereo_converted_events = [
        parsivel_read_from_pickle(file)
        for file in sorted(stereo_converted_events_folder.iterdir())
    ]
    print("Done.")

    print_general_info(parsivel_events, stereo_events, stereo_converted_events)
