from datetime import datetime
from datetime import timedelta

PARISOFFSET = 2 * 3600

"""
Function for reading the standard date format string and creating a datetime obj
"""


def standard_to_dtime(standard_date: int) -> datetime:
    dt_obj = datetime.strptime(str(standard_date), "%Y%m%d%H%M%S")
    return dt_obj


def standard_to_tstamp(standard_date: int) -> int:
    dt_obj = datetime.strptime(str(standard_date), "%Y%m%d%H%M%S")
    return int(dt_obj.timestamp() + PARISOFFSET)


def dt_to_tstamp(dtime: datetime) -> int:
    return int(dtime.timestamp() + PARISOFFSET)


def tstamp_to_readable(tstamp: int) -> str:
    dt_obj = datetime.utcfromtimestamp(tstamp)
    return dt_obj.strftime("%Y/%m/%d  %H:%M:%S")


"""
Functions for creating time ranges there are used for reading files
"""


def range_dtime_30s(beg: datetime, end: datetime):
    beg = beg.replace(second=(beg.second // 30) * 30)
    end = end.replace(second=(beg.second // 30) * 30)
    thirty_seconds = timedelta(0, 30)
    while beg < end:
        yield beg
        beg = beg + thirty_seconds


def range_dtime_1d(beg: datetime, end: datetime):
    beg = beg.replace(second=0, hour=0, minute=0)
    end = end.replace(second=0, hour=0, minute=0)
    one_day = timedelta(1, 0)
    while beg < end:
        yield beg
        beg = beg + one_day


"""
Function that reads the time from an integer in the format "%Y%m%d%H%M%S", rounds
to the clossest point on the set that start at the beggining of the day and has
the interval of the resolution given
Ex: resolution 30s
(30, 60 ... 86370)
"""


def standard_to_rounded_tstamp(standard_date: int, resolution: int) -> int:
    dt_obj = datetime.strptime(str(standard_date), "%Y%m%d%H%M%S")
    beg_day_tstamp = dt_obj.replace(hour=0, minute=0, second=0).timestamp()
    return int(
        (dt_obj.timestamp() - beg_day_tstamp) // resolution * resolution
        + beg_day_tstamp
        + PARISOFFSET
    )


def tstamp_to_standard(tstamp: int) -> int:
    dt_obj = datetime.utcfromtimestamp(tstamp)
    return int(dt_obj.strftime("%Y%m%d%H%M%S"))
