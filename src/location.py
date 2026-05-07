"""Location metadata generation for the Grand St. Bernard deployment.

The SensorScope ``stbernard-location.zip`` archive ships a station coordinate
table in the local Swiss CH1903 / LV03 projection ("station_gsb_XY.txt").
The current task only requires deployment-level coordinates in WGS84, which
are taken from the well-documented Grand St. Bernard pass location and
recorded in :mod:`src.paths`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import paths

logger = logging.getLogger(__name__)


def build_location_metadata(coordinate_source_file: Path | None = None) -> dict:
    """Assemble the Grand St. Bernard location-metadata payload.

    Args:
        coordinate_source_file: Optional path to the station coordinate file
            shipped with the dataset, recorded for traceability.

    Returns:
        Dictionary suitable for serialization.
    """
    return {
        "deployment_name": paths.GSB_DEPLOYMENT_NAME,
        "latitude": paths.GSB_LATITUDE,
        "longitude": paths.GSB_LONGITUDE,
        "altitude_m": paths.GSB_ALTITUDE_M,
        "timezone": paths.GSB_TIMEZONE,
        "coordinate_level": "deployment",
        "coordinate_source": (
            "Documented Grand St. Bernard pass coordinates (WGS84). "
            "The dataset's station_gsb_XY.txt provides per-station offsets "
            "in the Swiss CH1903 / LV03 projection but no direct WGS84 "
            "translation; deployment-level coordinates are used here."
        ),
        "station_coordinate_file": (
            str(coordinate_source_file) if coordinate_source_file else None
        ),
        "uncertainty_notes": (
            "Latitude/longitude refer to the pass and are accurate to within "
            "~1 km, sufficient for sunrise time computation (~4 s of arc per "
            "km in latitude)."
        ),
    }


def write_location_metadata(payload: dict,
                            out_path: Path = paths.LOCATION_METADATA_JSON
                            ) -> Path:
    """Persist location metadata to JSON."""
    paths.ensure_dirs()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Location metadata written to %s", out_path)
    return out_path
