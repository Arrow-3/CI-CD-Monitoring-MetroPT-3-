from unittest.mock import MagicMock, patch
from metropt.config import settings
from metropt.dtos import ProcessedRawDataDTO, ShiftEventDTO, ModelUpdateCompleteDTO
from metropt.services.transfer_learning import TransferLearningService


def _proc(sec: int, label: int = 0) -> str:
    return ProcessedRawDataDTO(
        timestamp=f"2020-02-01T00:00:{sec % 60:02d}+00:00",
        sensor_values=[0.03, 8.9, 1.0],
        sensor_names=["Motor_current", "TP3", "COMP"],
        metadata={"product_id": "APU", "tool_id": "metro-1", "label": label},
        quality_flags={"missing_values": False, "out_of_range": False,
                       "schema_version": "v1.0"},
    ).to_json()


def _shift(severity: str = "critical", score: float = 0.5) -> str:
    return ShiftEventDTO(
        ts="2020-04-18T00:00:00+00:00", baseline_version="v",
        stream="features", score=score, severity=severity,
        window={"size": 500, "start_ts": "", "end_ts": "", "top_features": []},
    ).to_json()


@patch("metropt.services.transfer_learning.StorageManager")
@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer_multi")
def test_shift_below_threshold_ignored(_c, mock_prod, mock_sm):
    mock_prod.return_value = MagicMock()
    svc = TransferLearningService()
    svc.handle(_shift(severity="warn"), topic="shift_events")
    assert svc.producer.send.call_count == 0


@patch("metropt.services.transfer_learning.StorageManager")
@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer_multi")
def test_small_buffer_skips_retrain(_c, mock_prod, mock_sm):
    mock_prod.return_value = MagicMock()
    svc = TransferLearningService()
    # Only 10 rows in buffer; well below the 1000 minimum
    for i in range(10):
        svc.handle(_proc(i), topic="processed_raw_data")
    svc.handle(_shift(), topic="shift_events")
    assert svc.producer.send.call_count == 0


@patch("metropt.services.transfer_learning.StorageManager")
@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer_multi")
def test_cooldown_blocks_second_retrain(_c, mock_prod, mock_sm):
    mock_prod.return_value = MagicMock()
    svc = TransferLearningService()
    svc._last_retrain_ts = 1e18   # simulate "just retrained"
    for i in range(2000):
        svc.handle(_proc(i, label=1 if i % 20 == 0 else 0),
                   topic="processed_raw_data")
    svc.handle(_shift(), topic="shift_events")
    assert svc.producer.send.call_count == 0