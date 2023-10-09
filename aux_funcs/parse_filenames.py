import re
from collections import namedtuple
from datetime import datetime

Parser = namedtuple("Parser", ["prefix", "format", "sufix"])

"""
A function for extracting the structure of the naming system used.
Returns a Parser tuple, that contains the:
    - prefix -> What comes before the date
    - format -> The format used for giving the date
    - sufix -> Contains the file type
"""


def get_parser(file_name: str) -> Parser:
    # rest, filetype = file_name.strip("_no_data").split(".")
    rest, filetype = file_name.split(".")
    prefix, date, rest_date = re.split(r"(\d\d\d\d)", rest, maxsplit=1)
    date = "".join([date, rest_date])

    # Count the number of digits and the number of _ characters
    has_time = sum([1 for _ in re.finditer(r"\d", date)]) > 8
    has_separators = sum([1 for _ in re.finditer("_", date)])
    date_format = "%Y%m%D%H%M%S"

    match (has_time, has_separators):
        case (True, 1):
            date_format = "%Y%m%d_%H%M%S"
        case (True, 0):
            date_format = "%Y%m%d%H%M%S"
        case (False, 0):
            date_format = "%Y%m%d"
        case (True, _):
            date_format = "%Y_%-m_%-d%-H%-M%-S"
        case _:
            assert False, "Could not get the parser from the file"

    return Parser(prefix, date_format, f".{filetype}")


"""
Function for, extracting a date from a file name given the parser for that file
name.
"""


def parse_date(file_name: str, parser: str) -> datetime:
    prefix, date_format, sufix = parser
    date_str = file_name.strip(prefix).strip(sufix)
    if date_format == "%Y_%-m_%-d%-H%-M%-S":
        year, month, day, hour, minute, second = [int(t) for t in date_str.split("_")]
        return datetime(year, month, day, hour, minute, second)

    return datetime.strptime(date_str, date_format)


"""
Function for construct a file name from a parser (file name structure) and the
datetime desired.
"""


def construct_file_name(date: datetime, parser: Parser) -> str:
    prefix, format, sufix = parser
    if format == "%Y_%-m_%-d%-H%-M%-S":
        date_str = (
            f"{date.year}_{date.month}_{date.day}_"
            f"{date.hour}_{date.minute}_{date.second}"
        )
    else:
        date_str = date.strftime(format)

    return f"{prefix}{date_str}{sufix}"
