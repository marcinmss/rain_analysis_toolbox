from stereo import stereo_read_from_pickle as stereo_read_from_pickle
from pathlib import Path


if __name__ == "__main__":
    output_folder = Path(
        "/home/marcio/stage_project/data/saved_events/Set01/events/stereo_converted/"
    )
    stereo_folder = Path(
        "/home/marcio/stage_project/data/saved_events/Set01/events/stereo/"
    )
    stereo_events = [
        stereo_read_from_pickle(file_path)
        for file_path in sorted(stereo_folder.iterdir())
    ]
    print("Read the data for stereo.")

    stereo_converted_events = [event.convert_to_parsivel() for event in stereo_events]
    print("Converted all the data")
    # Clear the folders from previous use
    for file in output_folder.iterdir():
        file.unlink()

    # Save the events in there respective folders
    for i, event in enumerate(stereo_converted_events):
        event.to_pickle(output_folder / f"event{i+1:>02}.obj")

    print("DONE.")
