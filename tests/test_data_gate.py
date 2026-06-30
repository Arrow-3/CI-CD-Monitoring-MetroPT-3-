from unittest.mock import MagicMock, patch
from metropt.dtos import RawDataDTO, ProcessedRawDataDTO
from metropt.services.data_gate import DataGateService


def _send_through(svc: DataGateService, raw: RawDataDTO) -> ProcessedRawDataDTO:
    svc.handle(raw.to_json())
    payload = svc.producer.send.call_args.args[1]
    return ProcessedRawDataDTO.from_json(payload)


@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_clean_row_passes_through(_c, mock_prod):
    mock_prod.return_value = MagicMock()
    svc = DataGateService()
    raw = RawDataDTO(
        ts="2020-02-01T00:00:00+00:00", product_id="APU", tool_id="metro-1",
        raw={"TP2": 1.0, "Motor_current": 3.4, "COMP": 1.0},
    )
    out = _send_through(svc, raw)
    assert out.metadata["label"] == 0
    assert out.quality_flags["missing_values"] is False
    assert out.quality_flags["out_of_range"] is False


@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_out_of_range_is_clipped_and_flagged(_c, mock_prod):
    mock_prod.return_value = MagicMock()
    svc = DataGateService()
    raw = RawDataDTO(
        ts="2020-02-01T00:00:00+00:00", product_id="APU", tool_id="metro-1",
        raw={"Motor_current": 999.0},  # nonsense
    )
    out = _send_through(svc, raw)
    idx = out.sensor_names.index("Motor_current")
    assert out.sensor_values[idx] == 10.0  # clipped to range max
    assert "Motor_current" in out.quality_flags["out_of_range"]


@patch("metropt.services.base.kafka_utils.get_producer")
@patch("metropt.services.base.kafka_utils.get_consumer")
def test_label_is_one_in_pre_failure_window(_c, mock_prod):
    mock_prod.return_value = MagicMock()
    svc = DataGateService()
    # First reported failure starts 2020-04-18 00:00; pre-window default is 6h.
    raw = RawDataDTO(
        ts="2020-04-17T20:00:00+00:00", product_id="APU", tool_id="metro-1",
        raw={"TP2": 1.0},
    )
    out = _send_through(svc, raw)
    assert out.metadata["label"] == 1