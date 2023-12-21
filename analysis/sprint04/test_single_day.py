"""
This file is for testing the day that Auguste presented to Jerry as an example of a lot of drops in a day
 17 Nov 2023 01:21:59
"""

from pathlib import Path
from stereo3d import stereo3d_read_from_zips

beg = 20231117000000
end = 20231117235959

source_folder = Path("/home/marcio/stage_project/data/Daily_raw_data_3D_stereo/")

data = stereo3d_read_from_zips(beg, end, source_folder)

len(data)
