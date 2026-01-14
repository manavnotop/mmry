import datetime


def parse_datetime(dt: str | datetime.datetime) -> datetime.datetime:
    """
    Parse a datetime value and ensure it's timezone-aware.

    Args:
        dt: Either an ISO format string or a datetime object.

    Returns:
        A timezone-aware datetime object.
    """
    if isinstance(dt, str):
        parsed = datetime.datetime.fromisoformat(dt)
    else:
        parsed = dt

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)

    return parsed
