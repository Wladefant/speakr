"""
Datetime normalization helpers.

Backend timestamps — including meeting_date — are stored as naive UTC and
converted to the viewer's timezone client-side (parseServerInstant). These
helpers normalize incoming values to that convention.
"""

from datetime import datetime, timezone
from typing import Optional


def to_utc_naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to naive UTC.

    Aware datetimes are converted to UTC before the tzinfo is stripped.
    Naive datetimes are returned unchanged (assumed to be UTC already).
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt
