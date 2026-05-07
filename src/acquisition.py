"""Data acquisition for the SensorScope dataset.

Downloads the SensorScope archive from Zenodo (if missing), extracts it, and
catalogs the contents so that downstream stages can locate the Grand St.
Bernard deployment files without hard-coded paths.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import paths

logger = logging.getLogger(__name__)

# File extensions considered as candidate tabular data files.
DATA_EXTENSIONS = {".txt", ".csv", ".dat", ".tsv"}
# File names typically used for column / variable definitions.
METADATA_HINTS = (
    "readme",
    "column",
    "columns",
    "variables",
    "variable",
    "definition",
    "metadata",
    "info",
    "stations",
    "station",
)

# Tokens that hint at a Grand St. Bernard deployment directory.
GSB_TOKENS = (
    "grand-st-bernard",
    "grand_st_bernard",
    "grandstbernard",
    "grand-saint-bernard",
    "grand_saint_bernard",
    "saint-bernard",
    "saintbernard",
    "stbernard",
    "st-bernard",
    "st_bernard",
    "gsb",
)


@dataclass
class FileEntry:
    """Lightweight description of a file inside the dataset."""

    path: str
    size_bytes: int
    extension: str


def download_archive(url: str = paths.SENSORSCOPE_DOWNLOAD_URL,
                     destination: Path = paths.SENSORSCOPE_ZIP,
                     force: bool = False) -> Path:
    """Download the SensorScope archive from Zenodo if not already present.

    Args:
        url: Direct download URL of the archive.
        destination: Local file path where the archive will be saved.
        force: If True, re-download even if the file already exists.

    Returns:
        Path to the local archive file.
    """
    paths.ensure_dirs()
    if destination.exists() and destination.stat().st_size > 0 and not force:
        logger.info("Archive already present at %s (%d bytes)",
                    destination, destination.stat().st_size)
        return destination
    logger.info("Downloading %s -> %s", url, destination)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out:
        chunk = 1024 * 1024
        while True:
            data = response.read(chunk)
            if not data:
                break
            out.write(data)
    logger.info("Downloaded %d bytes", destination.stat().st_size)
    return destination


def extract_archive(archive: Path = paths.SENSORSCOPE_ZIP,
                    target_dir: Path = paths.SENSORSCOPE_EXTRACTED_DIR,
                    force: bool = False) -> Path:
    """Extract the SensorScope archive into a target directory.

    Args:
        archive: Path of the zip archive on disk.
        target_dir: Destination directory.
        force: If True, re-extract even if the directory is non-empty.

    Returns:
        Path to the extraction directory.
    """
    paths.ensure_dirs()
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    target_dir.mkdir(parents=True, exist_ok=True)
    has_content = any(target_dir.iterdir())
    if has_content and not force:
        logger.info("Extraction directory %s is non-empty; skipping.", target_dir)
        return target_dir
    logger.info("Extracting %s -> %s", archive, target_dir)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(target_dir)
    return target_dir


# Inner archives that must be extracted to access the Grand St. Bernard
# meteorological data and station coordinates.
GSB_INNER_ARCHIVES = {
    "stbernard-meteo.zip": "stbernard/meteo",
    "stbernard-location.zip": "stbernard/location",
    "stbernard-monitor.zip": "stbernard/monitor",
}


def extract_inner_archives(extract_root: Path = paths.SENSORSCOPE_EXTRACTED_DIR,
                           force: bool = False) -> dict[str, Path]:
    """Extract the per-deployment archives nested inside ``Sensorscope.zip``.

    Args:
        extract_root: Root directory where ``Sensorscope.zip`` was extracted.
        force: Re-extract even when the destination directories already exist.

    Returns:
        Mapping from inner archive name to its extraction directory.
    """
    out: dict[str, Path] = {}
    # Locate the directory holding the per-deployment zips.
    inner_root = extract_root / "Sensorscope"
    if not inner_root.exists():
        # Fall back to any direct child folder.
        children = [p for p in extract_root.iterdir() if p.is_dir()]
        inner_root = children[0] if children else extract_root
    for archive_name, rel_target in GSB_INNER_ARCHIVES.items():
        archive_path = inner_root / archive_name
        if not archive_path.exists():
            logger.warning("Inner archive missing: %s", archive_path)
            continue
        target = inner_root / rel_target
        target.mkdir(parents=True, exist_ok=True)
        has_content = any(target.iterdir())
        if has_content and not force:
            logger.info("Inner archive already extracted: %s", target)
            out[archive_name] = target
            continue
        logger.info("Extracting inner archive %s -> %s", archive_path, target)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target)
        out[archive_name] = target
    return out


def _walk_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _is_metadata_file(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in METADATA_HINTS)


def _looks_like_gsb(path_parts: Iterable[str]) -> bool:
    joined = "/".join(path_parts).lower()
    return any(tok in joined for tok in GSB_TOKENS)


def discover_deployments(extract_root: Path) -> dict[str, list[FileEntry]]:
    """Group dataset files by inferred deployment directory.

    The SensorScope Zenodo archive typically contains one folder per
    deployment. This function uses the top-level directory under the
    extraction root as the deployment identifier.

    Args:
        extract_root: Root directory containing the extracted archive.

    Returns:
        Mapping ``deployment_name -> list of FileEntry``.
    """
    deployments: dict[str, list[FileEntry]] = {}
    for fp in _walk_files(extract_root):
        rel = fp.relative_to(extract_root)
        parts = rel.parts
        deployment = parts[0] if len(parts) > 1 else "_root"
        deployments.setdefault(deployment, []).append(
            FileEntry(
                path=str(rel),
                size_bytes=fp.stat().st_size,
                extension=fp.suffix.lower(),
            )
        )
    return deployments


def detect_gsb_deployment(deployments: dict[str, list[FileEntry]]) -> str | None:
    """Return the name of the deployment that matches Grand St. Bernard."""
    for name in deployments:
        if _looks_like_gsb([name]):
            return name
    # Fallback: search inside file paths.
    for name, files in deployments.items():
        for fe in files:
            if _looks_like_gsb(fe.path.split("/")):
                return name
    return None


def get_gsb_meteo_dir(extract_root: Path = paths.SENSORSCOPE_EXTRACTED_DIR
                      ) -> Path | None:
    """Return the directory containing Grand St. Bernard meteo files."""
    candidates = [
        extract_root / "Sensorscope" / "stbernard" / "meteo",
        extract_root / "stbernard" / "meteo",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("stbernard-meteo-*.txt")):
            return c
    # Generic search.
    for p in extract_root.rglob("stbernard-meteo-*.txt"):
        return p.parent
    return None


def get_gsb_location_dir(extract_root: Path = paths.SENSORSCOPE_EXTRACTED_DIR
                         ) -> Path | None:
    """Return the directory containing Grand St. Bernard station coordinates."""
    candidates = [
        extract_root / "Sensorscope" / "stbernard" / "location",
        extract_root / "stbernard" / "location",
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in extract_root.rglob("station_gsb_XY.txt"):
        return p.parent
    return None


def build_inventory(extract_root: Path = paths.SENSORSCOPE_EXTRACTED_DIR) -> dict:
    """Build a JSON-serializable inventory of the SensorScope dataset.

    Args:
        extract_root: Root directory containing the extracted archive.

    Returns:
        Dictionary describing the dataset structure.
    """
    deployments = discover_deployments(extract_root)
    gsb_name = detect_gsb_deployment(deployments)
    # Prefer the canonical deployment label whenever the meteo dir was found.
    if get_gsb_meteo_dir(extract_root) is not None:
        gsb_name = paths.GSB_DEPLOYMENT_NAME

    def _rel(p) -> str:
        try:
            return str(Path(p).resolve().relative_to(paths.ROOT_DIR))
        except Exception:
            return str(p)

    deployment_summaries = []
    for dep_name, files in sorted(deployments.items()):
        data_files = [f for f in files if f.extension in DATA_EXTENSIONS]
        metadata_files = [f for f in files if _is_metadata_file(Path(f.path).name)]
        deployment_summaries.append({
            "deployment_name": dep_name,
            "file_count": len(files),
            "data_file_count": len(data_files),
            "metadata_file_count": len(metadata_files),
            "total_size_bytes": sum(f.size_bytes for f in files),
            "metadata_files": [_rel(f.path) for f in metadata_files][:50],
            "candidate_data_files": [_rel(f.path) for f in data_files][:200],
            "all_files": [
                {"path": _rel(f.path), "size_bytes": f.size_bytes,
                 "extension": f.extension}
                for f in files[:500]
            ],
            "is_grand_st_bernard": dep_name == gsb_name,
        })

    inventory = {
        "dataset_name": "SensorScope environmental monitoring dataset",
        "source_url": paths.ZENODO_RECORD_URL,
        "citation": (
            "SensorScope: EPFL environmental wireless sensor network. "
            "Zenodo record 2654726."
        ),
        "local_raw_path": _rel(paths.SENSORSCOPE_RAW_DIR),
        "local_extracted_path": _rel(extract_root),
        "archive_path": _rel(paths.SENSORSCOPE_ZIP),
        "archive_size_bytes": (
            paths.SENSORSCOPE_ZIP.stat().st_size
            if paths.SENSORSCOPE_ZIP.exists() else None
        ),
        "acquisition_timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_deployment": gsb_name,
        "selected_deployment_basis": (
            "name match against Grand St. Bernard tokens" if gsb_name else None
        ),
        "gsb_meteo_dir": (
            _rel(get_gsb_meteo_dir(extract_root))
            if get_gsb_meteo_dir(extract_root) else None
        ),
        "gsb_location_dir": (
            _rel(get_gsb_location_dir(extract_root))
            if get_gsb_location_dir(extract_root) else None
        ),
        "deployments": deployment_summaries,
        "notes": [],
    }
    if not gsb_name:
        inventory["notes"].append(
            "Grand St. Bernard deployment could not be auto-detected."
        )
    if not deployments:
        inventory["notes"].append(
            "No files were found in the extraction directory."
        )
    return inventory


def write_inventory(inventory: dict,
                    out_path: Path = paths.DATASET_INVENTORY_JSON) -> Path:
    """Persist the inventory as JSON.

    Args:
        inventory: Inventory dictionary.
        out_path: Destination path for the JSON file.

    Returns:
        Path to the written JSON file.
    """
    paths.ensure_dirs()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(inventory, fh, indent=2, ensure_ascii=False)
    logger.info("Inventory written to %s", out_path)
    return out_path


def acquire(force_download: bool = False, force_extract: bool = False) -> dict:
    """Run the full acquisition pipeline.

    Args:
        force_download: Re-download even if the archive already exists.
        force_extract: Re-extract even if the directory is non-empty.

    Returns:
        The inventory dictionary.
    """
    paths.ensure_dirs()
    download_archive(force=force_download)
    extract_archive(force=force_extract)
    extract_inner_archives(force=force_extract)
    inventory = build_inventory()
    write_inventory(inventory)
    return inventory
