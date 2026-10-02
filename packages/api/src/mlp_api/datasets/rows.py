import json
from pathlib import Path

from mlp_core import config


def row_format_of_file(path: Path) -> str:
    """The one row format of every row in the JSONL file; ValueError naming the first bad row."""
    row_format = None
    with path.open("rb") as file:
        for number, line in enumerate(file, 1):
            try:
                line_format = row_format_of_line(line)
            except ValueError as error:
                raise ValueError(f"line {number}: {error}") from None
            if row_format not in (None, line_format):
                raise ValueError(
                    f"line {number}: is {line_format} rows, but line 1 is {row_format} rows"
                )
            row_format = line_format
    if row_format is None:
        raise ValueError("the file has no rows")
    return row_format


def row_format_of_line(line: bytes) -> str:
    try:
        row = json.loads(line)
    except ValueError:
        raise ValueError("is not JSON") from None
    if not isinstance(row, dict):
        raise ValueError("is not a JSON object")
    for row_format, fields in config.ROW_FORMATS.items():
        if fields.keys() <= row.keys():
            for field, value in fields.items():
                if not VALUE_CHECKS[value](row[field]):
                    raise ValueError(f"`{field}` must be {value}")
            return row_format
    raise ValueError(f"has the fields of no row format ({', '.join(config.ROW_FORMATS)})")


def is_messages(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in value)
    )


def is_list_of(kind: type):
    return lambda value: isinstance(value, list) and all(isinstance(v, kind) for v in value)


VALUE_CHECKS = {
    "a string": lambda value: isinstance(value, str),
    "a string or messages": lambda value: isinstance(value, str) or is_messages(value),
    "messages": is_messages,
    "true or false": lambda value: isinstance(value, bool),
    "a list of strings": is_list_of(str),
    "a list of true or false": is_list_of(bool),
}
