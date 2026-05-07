"""Sunrise / sunset astronomical ground truth using ``astral``."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun

from . import paths

logger = logging.getLogger(__name__)


def compute_sun_times(start_date: date, end_date: date,
                      latitude: float, longitude: float,
                      timezone_name: str,
                      include_sunset: bool = True
                      ) -> list[dict]:
    """Compute sunrise (and optionally sunset) for a date range.

    Args:
        start_date: First date (inclusive).
        end_date: Last date (inclusive).
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.
        timezone_name: IANA timezone name (e.g., ``"Europe/Zurich"``).
        include_sunset: When True, also report sunset times.

    Returns:
        One record per date, with ``sunrise_local``, ``sunrise_utc``, and
        optionally ``sunset_local`` and ``sunset_utc``.
    """
    tz = ZoneInfo(timezone_name)
    location = LocationInfo(name="site", region="region",
                            timezone=timezone_name,
                            latitude=latitude, longitude=longitude)
    out: list[dict] = []
    d = start_date
    while d <= end_date:
        s = sun(location.observer, date=d, tzinfo=tz)
        record = {
            "date": d.isoformat(),
            "sunrise_local": s["sunrise"].isoformat(),
            "sunrise_utc": s["sunrise"].astimezone(ZoneInfo("UTC")).isoformat(),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone_name,
            "method": "astral.sun.sun",
        }
        if include_sunset:
            record["sunset_local"] = s["sunset"].isoformat()
            record["sunset_utc"] = s["sunset"].astimezone(
                ZoneInfo("UTC")).isoformat()
        out.append(record)
        d += timedelta(days=1)
    return out


def build_ground_truth_for_dates(dates: list[str],
                                 latitude: float, longitude: float,
                                 timezone_name: str
                                 ) -> dict:
    """Build a sunrise ground-truth payload for a list of explicit dates.

    Args:
        dates: List of ISO date strings.
        latitude: Site latitude.
        longitude: Site longitude.
        timezone_name: IANA timezone name.

    Returns:
        Dictionary with ``records`` and global metadata fields.
    """
    if not dates:
        return {
            "records": [],
            "method": "astral.sun.sun",
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone_name,
            "notes": ["No valid dates provided."],
        }
    parsed = sorted({date.fromisoformat(d) for d in dates})
    records = compute_sun_times(parsed[0], parsed[-1], latitude, longitude,
                                timezone_name)
    keep = {d.isoformat() for d in parsed}
    records = [r for r in records if r["date"] in keep]
    return {
        "records": records,
        "method": "astral.sun.sun",
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_name,
        "n_dates": len(records),
        "date_range": {
            "start": records[0]["date"] if records else None,
            "end": records[-1]["date"] if records else None,
        },
    }


def write_ground_truth(payload: dict,
                       out_path: Path = paths.SUNRISE_GROUND_TRUTH_JSON
                       ) -> Path:
    """Persist the sunrise ground-truth payload to JSON."""
    paths.ensure_dirs()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Sunrise ground truth written to %s", out_path)
    return out_path
