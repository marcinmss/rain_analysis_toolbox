from datetime import datetime
from datetime import timedelta, timezone
from typing import Tuple
from numpy import ceil

PARISOFFSET = timezone(timedelta(hours=2))

"""
Function for reading the standard date format string and creating a datetime obj
"""


def standard_to_dtime(standard_date: int) -> datetime:
    dt_obj = datetime.strptime(str(standard_date), "%Y%m%d%H%M%S").replace(
        tzinfo=PARISOFFSET
    )
    return dt_obj


def standard_to_tstamp(standard_date: int) -> int:
    return int(standard_to_dtime(standard_date).timestamp())


def dt_to_tstamp(dtime: datetime) -> int:
    return int(dtime.timestamp())


def tstamp_to_dt(tstamp: int) -> datetime:
    dt_obj = (
        datetime.utcfromtimestamp(tstamp)
        .replace(tzinfo=timezone.utc)
        .astimezone(PARISOFFSET)
    )
    return dt_obj


def tstamp_to_readable(tstamp: int) -> str:
    dt_obj = tstamp_to_dt(tstamp)
    return dt_obj.strftime("%Y/%m/%d  %H:%M:%S")


"""
Functions for creating time ranges there are used for reading files
"""


def round_to_30s(tstamp: int) -> int:
    dt_obj = tstamp_to_dt(tstamp)
    return dt_to_tstamp(dt_obj.replace(second=(dt_obj.second // 30) * 30))


def round_startfinish_to_30s(
    start: datetime, finish: datetime
) -> Tuple[datetime, datetime]:
    start = start - timedelta(seconds=start.second % 30)
    finish = finish + timedelta(seconds=ceil(finish.second / 30) * 30 - finish.second)

    return (start, finish)


def range_dtime_30s(beg: datetime, end: datetime):
    beg, end = round_startfinish_to_30s(beg, end)
    thirty_seconds = timedelta(0, 30)
    while beg < end:
        yield beg
        beg = beg + thirty_seconds


def range_dtime_1d(beg: datetime, end: datetime):
    beg = beg.replace(second=0, hour=0, minute=0)
    end = end.replace(second=0, hour=0, minute=0)
    assert beg < end, "Beggining needs to come before the end"
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


# # def standard_to_rounded_tstamp(standard_date: int, resolution: int) -> int:
# #     dt_obj = datetime.strptime(str(standard_date), "%Y%m%d%H%M%S")
# #     beg_day_tstamp = dt_obj.replace(hour=0, minute=0, second=0).timestamp()
# #     return int(
# #         (dt_obj.timestamp() - beg_day_tstamp) // resolution * resolution
# #         + beg_day_tstamp
# #         + PARISOFFSET
# #     )
#
#
# def tstamp_to_standard(tstamp: int) -> int:
#     dt_obj = datetime.utcfromtimestamp(tstamp)
#     return int(dt_obj.strftime("%Y%m%d%H%M%S"))
