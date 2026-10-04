"""Central configuration for the predictive maintenance project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
DOCS_DIR = ROOT / "docs"

SEED = 42
N_MACHINES = 20          # number of simulated machines
N_DAYS = 180             # length of the simulated history (hourly readings)
START = "2026-01-01"

HORIZON_H = 24           # target: will the machine fail within the next 24 hours?
SENSORS = ["temperature", "vibration", "pressure"]
HISTORY_DAYS = 14        # days of history shown in the dashboards
RECALL_TARGET = 0.80     # we want to catch at least 80 % of the risky hours
