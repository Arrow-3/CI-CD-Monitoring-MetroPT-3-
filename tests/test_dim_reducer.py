from unittest.mock import MagicMock, patch
import numpy as np
from metropt.config import settings
from metropt.dtos import FeatureVectorDTO, Coordinates2DDTO
from metropt.services.dim_reducer import DimReducerService


def _fv(seed: int) -> str:
    rng = np.random.default_rng(seed)
    return FeatureVectorDTO(
        ts=f"2020-02-01T00:00:{seed % 60:02d}+00:00", product_id="APU",
        fe_version="v1.0",
        features={f"f{i}": float(rng.normal()) for i in range(10)},
    ).to_json()


@patch("metropt.services.dim_reducer.StorageManager")
@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_silent_during_bootstrap(_c, mock_prod, mock_sm):
    mock_prod.return_value = MagicMock()
    mock_sm.return_value.latest_model_version.return_value = None
    svc = DimReducerService()
    for i in range(settings.DR_BOOTSTRAP_SIZE - 1):
        svc.handle(_fv(i))
    assert svc.producer.send.call_count == 0
    assert svc.reducer is None


@patch("metropt.services.dim_reducer.StorageManager")
@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_fits_and_starts_publishing_at_bootstrap_end(_c, mock_prod, mock_sm):
    mock_prod.return_value = MagicMock()
    mock_sm.return_value.latest_model_version.return_value = None
    mock_sm.return_value.save_model.return_value = "/tmp/fake.joblib"
    svc = DimReducerService()

    for i in range(settings.DR_BOOTSTRAP_SIZE):
        svc.handle(_fv(i))
    assert svc.reducer is not None
    assert svc.producer.send.call_count == 1   # the trigger sample also emitted

    svc.handle(_fv(9999))
    assert svc.producer.send.call_count == 2

    payload = svc.producer.send.call_args.args[1]
    coord = Coordinates2DDTO.from_json(payload)
    assert isinstance(coord.x, float) and isinstance(coord.y, float)