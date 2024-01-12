from stereo3d import stereo_read_from_pickle as std_read
from parsivel import pars_read_from_pickle as pars_read
from aux_funcs.extract_events import is_event
from pathlib import Path
from matplotlib import pyplot as plt
from numpy import logical_and
import numpy as np


output_folder_parsivel = Path(
    "/home/marcio/stage_project/data/saved_events/sprint04/parsivel/"
)
output_folder_stereo = Path(
    "/home/marcio/stage_project/data/saved_events/sprint04/stereo/"
)


# The longest continuous period for both data
pars_full_event = Path("/home/marcio/stage_project/data/saved_events/parsivel_full.obj")
str_full_event = Path("/home/marcio/stage_project/data/saved_events/stereo_full.obj")

# Read the main parsivel series
parsivel_full_event = pars_read(pars_full_event)
stereo_full_event = std_read(str_full_event)
print("THE DATA FOR BOTH EVENTS WAS READ.")

# Find events for that are detected both in the parsivel and stereo devices
parsivel_events = is_event(parsivel_full_event.rain_rate, 15, 0.1)
stereo_events = is_event(stereo_full_event.rain_rate(), 15, 0.1)
simutainous_events = logical_and(parsivel_events, stereo_events)

# Colect the detected events into parsivel time series and plot to look at them
parsivel_events = parsivel_full_event.colect_events(simutainous_events)
print(f"Total of {len(parsivel_events)} events detected.")
n = int(np.ceil(len(parsivel_events) ** 0.5))
figure = plt.figure()
figure.set_size_inches((n * 3 + 3, n * 3 + 3))
figure.set_layout_engine("constrained")
for i, event in enumerate(parsivel_events):
    ax = figure.add_subplot(n, n, i + 1)
    ax.plot(event.rain_rate)
plt.show(block=False)


# Colect the events for the Stereo 3D as well and check to see is the durations match
stereo_events = stereo_full_event.extract_events(
    [pars_event.duration for pars_event in parsivel_events]
)
all(
    pars_event.duration == stereo_event.duration
    for pars_event, stereo_event in zip(parsivel_events, stereo_events)
)

# Filter the events by the mimimum legth we want
minimum_length = 2**6
stereo_events = [
    event
    for event, pars_event in zip(stereo_events, parsivel_events)
    if len(pars_event) >= minimum_length
]
parsivel_events = [event for event in parsivel_events if len(event) >= minimum_length]
print(f"OF THOSE, {len(parsivel_events)} HAVE THE MINIMAL LENGTH.({minimum_length})")

# Filter for outliers
outliers = [
    i
    for i, (pars_event, stereo_event) in enumerate(zip(parsivel_events, stereo_events))
    if pars_event.total_depth_for_event > 10000
    or stereo_event.total_depth_for_event > 10000
]
print("FILTERING FOR OUTLIERS")
print(f"     - TOTAL OF {len(outliers)} FOUND.")
print(f"     - THE OUTLIERS WHERE REMOVED.")
for i, idx in enumerate(outliers):
    parsivel_events.pop(idx - i)
    stereo_events.pop(idx - i)

# Clear the folders from previous use
for file in output_folder_parsivel.iterdir():
    file.unlink()
for file in output_folder_stereo.iterdir():
    file.unlink()

# Save the events in there respective folders
for i, (pars_event, stereo_event) in enumerate(zip(parsivel_events, stereo_events)):
    pars_event.to_pickle(output_folder_parsivel / f"event{i+1:>02}.obj")
    stereo_event.to_pickle(output_folder_stereo / f"event{i+1:>02}.obj")

print("ALL THE EVENTS DETECTED WERE SAVED.")
