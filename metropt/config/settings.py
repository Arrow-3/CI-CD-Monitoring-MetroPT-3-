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
