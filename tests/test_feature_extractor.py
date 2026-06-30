from unittest.mock import MagicMock, patch
import pandas as pd
from metropt.dtos import ProcessedRawDataDTO, FeatureVectorDTO
from metropt.services.feature_extractor import FeatureExtractorService


def _msg(ts: str, motor: float = 0.035, tp3: float = 8.9) -> str:
    return ProcessedRawDataDTO(
        timestamp=ts,
        sensor_values=[motor, tp3, 1.0],
        sensor_names=["Motor_current", "TP3", "COMP"],
        metadata={"product_id": "APU", "tool_id": "metro-1", "label": 0},
        quality_flags={"missing_values": False, "out_of_range": False,
                       "schema_version": "v1.0"},
    ).to_json()


@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_bootstrap_period_emits_nothing(_c, mock_prod):
    mock_prod.return_value = MagicMock()
    svc = FeatureExtractorService()
    svc.handle(_msg("2020-02-01T00:00:00+00:00"))
    svc.handle(_msg("2020-02-01T00:00:05+00:00"))
    assert svc.producer.send.call_count == 0  # still inside the 30s window


@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_emits_after_window_fills(_c, mock_prod):
    mock_prod.return_value = MagicMock()
    svc = FeatureExtractorService()
    # 31 seconds of synthetic stream
    for sec in range(31):
        svc.handle(_msg(f"2020-02-01T00:00:{sec:02d}+00:00",
                        motor=0.03 + 0.001 * sec))
    assert svc.producer.send.call_count >= 1
    payload = svc.producer.send.call_args.args[1]
    dto = FeatureVectorDTO.from_json(payload)

    # 7 analog × 4 stats + 1 digital pass-through (COMP)
    assert "Motor_current_mean" in dto.features
    assert "Motor_current_std" in dto.features
    assert dto.features["Motor_current_std"] > 0  # values were varying
    assert dto.features["COMP"] == 1.0            # digital passthrough


@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_old_values_evicted_from_window(_c, mock_prod):
    mock_prod.return_value = MagicMock()
    svc = FeatureExtractorService()
    # First batch at t=0..30 with motor=1.0
    for sec in range(31):
        svc.handle(_msg(f"2020-02-01T00:00:{sec:02d}+00:00", motor=1.0))
    # Jump forward 60s with motor=5.0 — old values should be gone
    for sec in range(31):
        svc.handle(_msg(f"2020-02-01T00:02:{sec:02d}+00:00", motor=5.0))

    payload = svc.producer.send.call_args.args[1]
    dto = FeatureVectorDTO.from_json(payload)
    assert abs(dto.features["Motor_current_mean"] - 5.0) < 1e-6  # purely new data