from __future__ import annotations

from datetime import datetime, time, timezone


def years_ago(dt: datetime, years: int) -> datetime:
    try:
        return dt.replace(year=dt.year - years)
    except ValueError:
        return dt.replace(year=dt.year - years, month=2, day=28)


def parse_date_arg(
    date_string: str | None,
    default: datetime,
    end_of_day: bool = False,
) -> datetime:
    if date_string is None:
        return default

    parsed_date = datetime.fromisoformat(date_string)

    if parsed_date.tzinfo is None:
        if "T" not in date_string and end_of_day:
            parsed_date = datetime.combine(
                parsed_date.date(),
                time(hour=23, minute=59, second=59),
                tzinfo=timezone.utc,
            )
        else:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)

    return parsed_date