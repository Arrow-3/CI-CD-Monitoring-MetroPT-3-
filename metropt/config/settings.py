from __future__ import annotations
import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
NUM_PARTITIONS = int(os.getenv("PRISM_NUM_PARTITIONS", "3"))
REPLICATION_FACTOR = 1

TOPICS = {
    "raw_data_stream", "processed_raw_data", "feature_vectors",
    "predictions", "coordinates_nd", "shift_events", "model_update_complete",
}

def get_consumer_group_id(service_name: str) -> str:
    return f"metropt_{service_name.lower()}"

# --- Data replay ---
DATA_PATH = os.getenv("PRISM_DATA_PATH", "C:\Drive W\Machine Learning\MetroPT -3\MetroPT3(CompressorDatase).csv")
REPLAY_SPEED = float(os.getenv("PRISM_REPLAY_SPEED", "600"))  # 600x → ~6h of data per real minute

# Sensor columns. VERIFY against your CSV header (explore_data prints it).
# Names differ slightly between the MetroPT and MetroPT-3 releases.
ANALOG_COLS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
               "Oil_temperature", "Motor_current"]
DIGITAL_COLS = ["COMP", "DV_eletric", "Towers", "MPG", "LPS",
                "Pressure_switch", "Oil_level", "Caudal_impulses"]

# --- Labeling: reported air-leak failures (CONFIRM against the reference paper) ---
FAILURE_WINDOWS = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]
PRE_FAILURE_HOURS = 6

SCHEMA_VERSION = "v1.0"

# Validation ranges per analog sensor: (min, max).
# CALIBRATE these from explore_data output, not from these defaults.
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "TP2":             (-1.0,  12.0),
    "TP3":             (-1.0,  12.0),
    "H1":              (-1.0,  12.0),
    "DV_pressure":     (-1.0,  12.0),
    "Reservoirs":      (-1.0,  12.0),
    "Oil_temperature": ( 0.0, 100.0),
    "Motor_current":   ( 0.0,  10.0),
}
# Digital signals: must be 0 or 1.
DIGITAL_VALID = {0.0, 1.0}  


# --- FeatureExtractor ---
FE_VERSION = "v1.0"
FE_WINDOW_SECONDS = 30
# Which sensors get rolling stats. Digital signals don't — mean/std of 0/1
# flags isn't meaningful here. (Counts of flips would be, but that's v2.)
FE_ANALOG_FEATURES = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                      "Oil_temperature", "Motor_current"]
# Statistics to compute per analog sensor.
FE_STATS = ["mean", "std", "min", "max"]



# --- PrimaryModel ---
PROJECT = "metropt"
MODEL_TYPE = "primary"
TRAIN_SAMPLE_LIMIT = 200_000        # cap rows for tractable training
TRAIN_TEST_SPLIT = 0.2
MODEL_PRED_THRESHOLD = 0.5          # prob ≥ threshold → label=1