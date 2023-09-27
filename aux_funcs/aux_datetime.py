from datetime import datetime


def standard_to_tstamp(standard_date: int) -> int:
    dt_obj = datetime.strptime(str(standard_date), "%Y%m%d%H%M%S")
    return int(dt_obj.timestamp() + PARISOFFSET)


"""
Function that reads the time from an integer in the format "%Y%m%d%H%M%S", rounds
to the clossest point on the set that start at the beggining of the day and has
the interval of the resolution given
Ex: resolution 30s
(30, 60 ... 86370)
"""
PARISOFFSET = 2 * 3600


def standard_to_rounded_tstamp(standard_date: int, resolution: int) -> int:
    dt_obj = datetime.strptime(str(standard_date), "%Y%m%d%H%M%S")
    beg_day_tstamp = dt_obj.replace(hour=0, minute=0, second=0).timestamp()
    return int(
        (dt_obj.timestamp() - beg_day_tstamp) // resolution * resolution
        + beg_day_tstamp
        + PARISOFFSET
    )


def tstamp_to_readable(tstamp: int) -> str:
    dt_obj = datetime.utcfromtimestamp(tstamp)
    return dt_obj.strftime("%Y/%m/%d  %H:%M:%S")


def tstamp_to_standard(tstamp: int) -> int:
    dt_obj = datetime.utcfromtimestamp(tstamp)
    return int(dt_obj.strftime("%Y%m%d%H%M%S"))
