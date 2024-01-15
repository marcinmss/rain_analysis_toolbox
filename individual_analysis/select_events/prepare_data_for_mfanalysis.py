from pathlib import Path
from stereo3d import stereo_read_from_pickle
from multifractal_analysis.data_prep import prep_data_ensemble
from numpy import concatenate, save

stereo_events_folder = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/events/stereo/"
)
OUTPUTFOlDER = Path("/home/marcio/stage_project/data/saved_events/Set01/")


def main():
    # Read the data from the stere 3D Events
    print("Reading events for 3D Stereo")
    events = [
        stereo_read_from_pickle(path) for path in sorted(stereo_events_folder.iterdir())
    ]
    print("    done.")

    print("Taking the rain rate for every event.")
    # Get the rain rate and delete the object
    list_rain_rate = [event.rain_rate(0.001) for event in events]
    del events
    print("    done.")

    # Prepare the data and concatenate it
    print("Preparing the data for the direct field")
    data = prep_data_ensemble(list_rain_rate[0], 2**6)
    for series in list_rain_rate[1:]:
        data = concatenate([data, prep_data_ensemble(series, 2**6)], axis=1)
    print("    done.")

    print("Saving the data for the direct field")
    save(OUTPUTFOlDER / "stereo_ensemble_direct_field_1ms_30min.npy", data)
    print("    done.")
    del data

    print("Preparing the data for the fluctuations")
    data_fluc = prep_data_ensemble(list_rain_rate[0], 2**6, fluc=True)
    for series in list_rain_rate[1:]:
        data_fluc = concatenate(
            [data_fluc, prep_data_ensemble(series, 2**6, fluc=True)], axis=1
        )
    print("    done.")

    print("Saving the data for the fluctuations")
    save(OUTPUTFOlDER / "stereo_ensemble_fluctuations_1ms_30min.npy", data_fluc)
    print("    done.")
    del data_fluc


if __name__ == "__main__":
    main()
