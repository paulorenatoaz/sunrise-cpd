"""Project-wide path constants for the Sunrise CPD project."""
from __future__ import annotations

from pathlib import Path

# Root directory of the project (parent of `src/`).
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Data directories.
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
SENSORSCOPE_RAW_DIR: Path = RAW_DIR / "sensorscope"
SENSORSCOPE_EXTRACTED_DIR: Path = SENSORSCOPE_RAW_DIR / "extracted"
SENSORSCOPE_ZIP: Path = SENSORSCOPE_RAW_DIR / "Sensorscope.zip"
EXTERNAL_DIR: Path = DATA_DIR / "external"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Output directories.
OUTPUT_DIR: Path = ROOT_DIR / "output"
JSON_DIR: Path = OUTPUT_DIR / "json"
REPORTS_DIR: Path = OUTPUT_DIR / "reports"
ASSETS_DIR: Path = REPORTS_DIR / "assets"

# JSON file paths.
DATASET_INVENTORY_JSON: Path = JSON_DIR / "dataset_inventory.json"
VARIABLE_SELECTION_JSON: Path = JSON_DIR / "variable_selection.json"
LOCATION_METADATA_JSON: Path = JSON_DIR / "location_metadata.json"
SUNRISE_GROUND_TRUTH_JSON: Path = JSON_DIR / "sunrise_ground_truth.json"
PREPROCESSING_SUMMARY_JSON: Path = JSON_DIR / "preprocessing_summary.json"
SENSOR_INFORMATIVENESS_JSON: Path = JSON_DIR / "sensor_informativeness.json"

# Processed data paths.
SYNCHRONIZED_PARQUET: Path = PROCESSED_DIR / "synchronized_sensor_data.parquet"
SYNCHRONIZED_CSV: Path = PROCESSED_DIR / "synchronized_sensor_data.csv"
SENSOR_MATRIX_PARQUET: Path = PROCESSED_DIR / "sensor_matrix.parquet"

# Report paths.
DATASET_REPORT_HTML: Path = REPORTS_DIR / "dataset_report.html"

# Dataset source.
ZENODO_RECORD_URL: str = "https://zenodo.org/records/2654726"
SENSORSCOPE_DOWNLOAD_URL: str = (
    "https://zenodo.org/records/2654726/files/Sensorscope.zip?download=1"
)

# Grand St. Bernard deployment metadata.
GSB_DEPLOYMENT_NAME: str = "Grand-St-Bernard"
GSB_LATITUDE: float = 45.8694
GSB_LONGITUDE: float = 7.1706
GSB_ALTITUDE_M: float = 2469.0
GSB_TIMEZONE: str = "Europe/Zurich"


def ensure_dirs() -> None:
    """Create all standard project directories if they do not exist."""
    for path in (
        RAW_DIR,
        SENSORSCOPE_RAW_DIR,
        SENSORSCOPE_EXTRACTED_DIR,
        EXTERNAL_DIR,
        PROCESSED_DIR,
        JSON_DIR,
        REPORTS_DIR,
        ASSETS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
