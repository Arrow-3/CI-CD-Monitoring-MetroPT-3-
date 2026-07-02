from unittest.mock import MagicMock, patch
import numpy as np
from metropt.config import settings
from metropt.dtos import FeatureVectorDTO, ShiftEventDTO
from metropt.services.distribution_monitor import DistributionMonitorService


def _fv(seed: int, shift: float = 0.0) -> str:
    rng = np.random.default_rng(seed)
    return FeatureVectorDTO(
        ts=f"2020-02-01T00:00:{seed % 60:02d}+00:00", product_id="APU",
        fe_version="v1.0",
        features={f"f{i}": float(rng.normal() + shift) for i in range(10)},
    ).to_json()


@patch("metropt.services.distribution_monitor.StorageManager")
@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_no_drift_no_events(_c, mock_prod, mock_sm):
    mock_prod.return_value = MagicMock()
    mock_sm.return_value.latest_model_version.return_value = None
    mock_sm.return_value.save_model.return_value = "/tmp/x.joblib"
    svc = DistributionMonitorService()

    # 1000 baseline + 500 identical-distribution current
    for i in range(settings.DM_BASELINE_SIZE + settings.DM_WINDOW_SIZE):
        svc.handle(_fv(i, shift=0.0))
    assert svc.producer.send.call_count == 0


@patch("metropt.services.distribution_monitor.StorageManager")
@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_large_shift_emits_event(_c, mock_prod, mock_sm):
    mock_prod.return_value = MagicMock()
    mock_sm.return_value.latest_model_version.return_value = None
    mock_sm.return_value.save_model.return_value = "/tmp/x.joblib"
    svc = DistributionMonitorService()

    # Baseline
    for i in range(settings.DM_BASELINE_SIZE):
        svc.handle(_fv(i, shift=0.0))
    # Current distribution shifted by 2.0 SDs
    for i in range(settings.DM_WINDOW_SIZE):
        svc.handle(_fv(10_000 + i, shift=2.0))

    assert svc.producer.send.call_count >= 1
    payload = svc.producer.send.call_args.args[1]
    event = ShiftEventDTO.from_json(payload)
    assert event.severity in {"warn", "alert", "critical"}
    assert event.score > settings.DM_THRESHOLDS["warn"]