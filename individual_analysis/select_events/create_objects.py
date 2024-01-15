"""
Read the data for both devices, turn them into an usable object and save that object for latter use
"""
from pathlib import Path
from parsivel import pars_read_from_zips
from stereo3d import stereo3d_read_from_zips


BEGGINING = 20230705000000
END = 20231202235959
OUTPUTFOLDER = Path("/home/marcio/stage_project/data/saved_events/Set01/")
stereo_source_folder = Path("/home/marcio/stage_project/data/Daily_raw_data_3D_stereo/")

parsivel_source_folder = Path("/home/marcio/stage_project/data/Pars_1/")


def main():
    print("Reading the data for Stereo 3D...")
    stereo_obj = stereo3d_read_from_zips(BEGGINING, END, stereo_source_folder)
    print("    done.")
    print("Read the data for Parsivel...")
    parsivel_obj = pars_read_from_zips(BEGGINING, END, parsivel_source_folder)
    print("    done.")

    # Save both objects
    print("Saving object for Stereo 3D...")
    stereo_obj.to_pickle(OUTPUTFOLDER / "stereo_full_event.obj")
    print("    done.")
    print("Saving object for Parsivel...")
    parsivel_obj.to_pickle(OUTPUTFOLDER / "parsivel_full_event.obj")
    print("    done.")


if __name__ == "__main__":
    main()
