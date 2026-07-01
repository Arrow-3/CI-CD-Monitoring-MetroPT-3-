from unittest.mock import MagicMock, patch
import numpy as np
from metropt.dtos import FeatureVectorDTO, PredictionDTO
from metropt.services.primary_model import PrimaryModelService


def _stub_model(prob: float = 0.9):
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - prob, prob]])
    return m


@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_predicts_when_model_loaded(_c, mock_prod):
    mock_prod.return_value = MagicMock()
    svc = PrimaryModelService()
    svc.model = _stub_model(prob=0.87)
    svc.feature_names = ["TP3_mean", "Motor_current_std"]
    svc.model_version = "20240101_000000"

    fv = FeatureVectorDTO(
        ts="2020-04-17T20:00:00+00:00", product_id="APU", fe_version="v1.0",
        features={"TP3_mean": 8.9, "Motor_current_std": 0.05, "extra_ignored": 99.0},
    )
    svc.handle(fv.to_json())

    payload = svc.producer.send.call_args.args[1]
    pred = PredictionDTO.from_json(payload)
    assert pred.prob == 0.87
    assert pred.label == 1            # ≥ 0.5 threshold
    assert pred.model_version == "20240101_000000"


@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_silent_when_no_model(_c, mock_prod):
    mock_prod.return_value = MagicMock()
    svc = PrimaryModelService()
    svc.model = None     # simulate cold start with no artifact yet
    svc.handle(FeatureVectorDTO(
        ts="2020-02-01T00:00:00+00:00", product_id="APU", fe_version="v1.0",
        features={"x": 1.0},
    ).to_json())
    assert svc.producer.send.call_count == 0