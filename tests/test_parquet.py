from pathlib import Path
from pandas import DataFrame
from pyarrow import schema, field as pa_field, table as patable
from stereo import stereo_read_from_pickle
from stereo.read_write import write_to_parquet, stereo_read_from_parquet

stereo_file = Path(
    "/home/marcio/stage_project/data/saved_events/Set01/stereo_full_event.obj"
)
outputfile = "/home/marcio/stage_project/mytoolbox/tests/test.parquet"

read_series = stereo_read_from_parquet(outputfile)


series = stereo_read_from_pickle(stereo_file)

write_to_parquet(outputfile, series)


read_series.total_depth_for_event()
