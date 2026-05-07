"""Dataset inspection: variable detection, timestamp probing, station counting.

The SensorScope archive ships data files with no fixed schema. Each row in a
station data file typically encodes a timestamp followed by sensor readings.
The ordering of the columns is documented in a separate "column definitions"
text file. This module parses both kinds of files and produces a structured
description of the deployment.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# Keywords used to score whether a variable is light-related, in priority order.
LIGHT_KEYWORDS_PRIORITY = [
    ("solar radiation", 100),
    ("solar_radiation", 100),
    ("incoming solar radiation", 100),
    ("incoming_solar_radiation", 100),
    ("solar", 90),
    ("radiation", 80),
    ("light", 70),
    ("luminosity", 70),
    ("lux", 65),
    ("par", 50),
    ("irradiance", 85),
    ("sunlight", 75),
]

# Tokens that strongly hint at a timestamp column.
TIMESTAMP_TOKENS = ("epoch", "time", "date", "timestamp", "unix")

# Tokens for station / sensor identifiers.
STATION_TOKENS = ("station", "node", "sensor_id", "sensorid", "id", "mote")


@dataclass
class ColumnDefinition:
    """One column entry parsed from a column-definitions metadata file."""

    index: int
    name: str
    raw_line: str = ""


@dataclass
class DeploymentInspection:
    """Result of inspecting a deployment directory."""

    deployment_name: str
    deployment_dir: Path
    column_definition_file: Path | None = None
    column_definitions: list[ColumnDefinition] = field(default_factory=list)
    data_files: list[Path] = field(default_factory=list)
    station_files: dict[str, Path] = field(default_factory=dict)
    delimiter: str | None = None
    n_columns_detected: int | None = None
    candidate_timestamp_columns: list[str] = field(default_factory=list)
    candidate_station_columns: list[str] = field(default_factory=list)
    candidate_light_columns: list[tuple[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable view."""
        return {
            "deployment_name": self.deployment_name,
            "deployment_dir": str(self.deployment_dir),
            "column_definition_file": (
                str(self.column_definition_file)
                if self.column_definition_file else None
            ),
            "n_columns_detected": self.n_columns_detected,
            "delimiter": self.delimiter,
            "n_data_files": len(self.data_files),
            "n_stations": len(self.station_files),
            "stations": sorted(self.station_files.keys()),
            "column_definitions": [
                {"index": c.index, "name": c.name} for c in self.column_definitions
            ],
            "candidate_timestamp_columns": self.candidate_timestamp_columns,
            "candidate_station_columns": self.candidate_station_columns,
            "candidate_light_columns": [
                {"name": n, "score": s} for n, s in self.candidate_light_columns
            ],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Column definitions parsing
# ---------------------------------------------------------------------------

_LINE_PATTERNS = [
    # "1: Station ID" or "1 - Station ID" or "1. Station ID"
    re.compile(r"^\s*(\d+)\s*[:\-\.\)]\s*(.+?)\s*$"),
    # "Column 1 : Station ID"
    re.compile(r"^\s*column\s*(\d+)\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE),
]


def parse_column_definitions(text: str) -> list[ColumnDefinition]:
    """Parse a column-definitions text file into structured entries.

    Args:
        text: Raw text content of the metadata file.

    Returns:
        List of column definitions sorted by column index.
    """
    defs: dict[int, ColumnDefinition] = {}
    for line in text.splitlines():
        for pat in _LINE_PATTERNS:
            m = pat.match(line)
            if m:
                idx = int(m.group(1))
                name = m.group(2).strip().rstrip(".;,")
                defs[idx] = ColumnDefinition(index=idx, name=name, raw_line=line)
                break
    return [defs[k] for k in sorted(defs)]


def find_column_definition_file(deployment_dir: Path,
                                fallback_dirs: list[Path] | None = None
                                ) -> Path | None:
    """Locate the column-definitions text file inside a deployment directory.

    When the deployment archive does not bundle a column-definitions file
    (this is the case for ``stbernard-meteo.zip`` on Zenodo), the search
    falls back to ``fallback_dirs``.
    """
    search_dirs: list[Path] = [deployment_dir]
    if fallback_dirs:
        search_dirs.extend(fallback_dirs)
    candidates: list[Path] = []
    for root in search_dirs:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            low = p.name.lower()
            if any(tok in low for tok in
                   ("column", "readme", "variable", "definition")):
                candidates.append(p)
    if not candidates:
        return None
    # Prefer the one that contains explicit numbered column entries.
    best: tuple[int, Path] | None = None
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        defs = parse_column_definitions(text)
        score = len(defs)
        if best is None or score > best[0]:
            best = (score, p)
    return best[1] if best else candidates[0]


# ---------------------------------------------------------------------------
# Variable scoring
# ---------------------------------------------------------------------------

def score_light_variable(name: str) -> int:
    """Return a relevance score for a candidate variable name.

    Higher is better. Zero means not light-related.
    """
    low = name.lower()
    score = 0
    for kw, val in LIGHT_KEYWORDS_PRIORITY:
        if kw in low:
            score = max(score, val)
    return score


def find_candidate_light_columns(
        defs: Iterable[ColumnDefinition]) -> list[tuple[str, int]]:
    """Score and rank columns by their likelihood of being light-related."""
    scored = []
    for cd in defs:
        s = score_light_variable(cd.name)
        if s > 0:
            scored.append((cd.name, s))
    scored.sort(key=lambda x: -x[1])
    return scored


def find_candidate_timestamp_columns(
        defs: Iterable[ColumnDefinition]) -> list[str]:
    """Identify columns whose names suggest a timestamp."""
    out = []
    for cd in defs:
        low = cd.name.lower()
        if any(tok in low for tok in TIMESTAMP_TOKENS):
            out.append(cd.name)
    return out


def find_candidate_station_columns(
        defs: Iterable[ColumnDefinition]) -> list[str]:
    """Identify columns whose names suggest a station / sensor identifier."""
    out = []
    for cd in defs:
        low = cd.name.lower()
        if any(tok in low for tok in STATION_TOKENS):
            out.append(cd.name)
    return out


# ---------------------------------------------------------------------------
# Data file detection
# ---------------------------------------------------------------------------

# Files like "1.txt", "station1.txt" or "stbernard-meteo-10.txt".
_STATION_NAME_PATTERNS = [
    re.compile(r"^stbernard-meteo-(\d+)\.txt$", re.IGNORECASE),
    re.compile(r"^[a-z]+-meteo-(\d+)\.txt$", re.IGNORECASE),
    re.compile(r"^(\d+)\.txt$", re.IGNORECASE),
    re.compile(r"^station[_\-]?(\d+)\.txt$", re.IGNORECASE),
    re.compile(r"^node[_\-]?(\d+)\.txt$", re.IGNORECASE),
]


def detect_station_files(deployment_dir: Path) -> dict[str, Path]:
    """Map station IDs to their corresponding data file paths.

    Args:
        deployment_dir: Directory of a single deployment.

    Returns:
        Mapping ``station_id -> file path``. Station IDs are strings.
    """
    out: dict[str, Path] = {}
    for p in deployment_dir.rglob("*"):
        if not p.is_file():
            continue
        for pat in _STATION_NAME_PATTERNS:
            m = pat.match(p.name)
            if m:
                out[m.group(1)] = p
                break
    return out


def sniff_delimiter(sample: str) -> str:
    """Return the most likely delimiter for a SensorScope data line."""
    # SensorScope station files are space-separated.
    counts = {
        "\t": sample.count("\t"),
        " ": sample.count(" "),
        ",": sample.count(","),
        ";": sample.count(";"),
    }
    return max(counts, key=counts.get) if any(counts.values()) else " "


def probe_data_file(path: Path, n_lines: int = 20) -> tuple[str, int]:
    """Probe a data file and return (delimiter, n_columns).

    Args:
        path: File to probe.
        n_lines: Number of lines to read.

    Returns:
        Tuple ``(delimiter, n_columns)``. ``n_columns`` may be 0.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = []
            for _ in range(n_lines):
                line = fh.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
    except OSError:
        return " ", 0
    if not lines:
        return " ", 0
    delim = sniff_delimiter(lines[0])
    if delim == " ":
        n_cols = len(re.split(r"\s+", lines[0].strip()))
    else:
        n_cols = len(lines[0].split(delim))
    return delim, n_cols


# ---------------------------------------------------------------------------
# Top-level inspection
# ---------------------------------------------------------------------------

def inspect_deployment(deployment_dir: Path,
                       deployment_name: str | None = None,
                       fallback_metadata_dirs: list[Path] | None = None
                       ) -> DeploymentInspection:
    """Inspect a deployment directory and return a structured summary.

    Args:
        deployment_dir: Path of the deployment directory.
        deployment_name: Optional explicit name; defaults to directory name.

    Returns:
        DeploymentInspection instance.
    """
    name = deployment_name or deployment_dir.name
    insp = DeploymentInspection(deployment_name=name,
                                deployment_dir=deployment_dir)

    cd_file = find_column_definition_file(
        deployment_dir, fallback_dirs=fallback_metadata_dirs)
    insp.column_definition_file = cd_file
    if cd_file:
        try:
            text = cd_file.read_text(encoding="utf-8", errors="replace")
            insp.column_definitions = parse_column_definitions(text)
        except OSError as exc:
            insp.notes.append(f"Could not read column file {cd_file}: {exc}")
    else:
        insp.notes.append("No column-definitions metadata file found.")

    insp.candidate_timestamp_columns = find_candidate_timestamp_columns(
        insp.column_definitions)
    insp.candidate_station_columns = find_candidate_station_columns(
        insp.column_definitions)
    insp.candidate_light_columns = find_candidate_light_columns(
        insp.column_definitions)

    insp.station_files = detect_station_files(deployment_dir)
    insp.data_files = list(insp.station_files.values())

    if insp.data_files:
        sample = insp.data_files[0]
        delim, n_cols = probe_data_file(sample)
        insp.delimiter = delim
        insp.n_columns_detected = n_cols
        if insp.column_definitions and n_cols and \
                len(insp.column_definitions) != n_cols:
            insp.notes.append(
                f"Column count mismatch: definitions={len(insp.column_definitions)}, "
                f"data file columns={n_cols} (sample: {sample.name})."
            )
    else:
        insp.notes.append("No station data files detected.")

    return insp


def select_best_light_variable(insp: DeploymentInspection
                               ) -> tuple[str | None, str]:
    """Select the highest-scoring light-related variable for a deployment.

    Args:
        insp: Inspection result.

    Returns:
        Tuple ``(variable_name, reason)``. ``variable_name`` is None when no
        candidate is available.
    """
    if not insp.candidate_light_columns:
        return None, "No light-related variable found among column definitions."
    name, score = insp.candidate_light_columns[0]
    return name, (
        f"Selected '{name}' with priority score {score} from candidates "
        f"{[n for n, _ in insp.candidate_light_columns]}."
    )
