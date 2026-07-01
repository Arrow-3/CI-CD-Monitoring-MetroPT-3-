from metropt.dtos import RawDataDTO

def test_rawdata_roundtrip():
    dto = RawDataDTO(ts="2020-04-18T00:00:00+00:00", product_id="APU",
                     tool_id="metro-1", raw={"TP2": 1.0, "Motor_current": 3.4})
    again = RawDataDTO.from_json(dto.to_json())
    assert again == dto
    assert again.to_ndarray().tolist() == [1.0, 3.4]


from metropt.dtos import ProcessedRawDataDTO

def test_processed_raw_roundtrip():
    dto = ProcessedRawDataDTO(
        timestamp="2020-02-01T00:00:00+00:00",
        sensor_values=[1.0, 3.4],
        sensor_names=["TP2", "Motor_current"],
        metadata={"product_id": "APU", "tool_id": "metro-1", "label": 0},
        quality_flags={"missing_values": False, "out_of_range": False,
                       "schema_version": "v1.0"},
    )
    again = ProcessedRawDataDTO.from_json(dto.to_json())
    assert again == dto


from metropt.dtos import FeatureVectorDTO

def test_feature_vector_roundtrip():
    dto = FeatureVectorDTO(
        ts="2020-02-01T00:00:00+00:00", product_id="APU",
        fe_version="v1.0",
        features={"Motor_current_mean": 0.035, "TP3_std": 0.012},
    )
    again = FeatureVectorDTO.from_json(dto.to_json())
    assert again == dto
    assert again.to_ndarray().tolist() == [0.035, 0.012]  # sorted keys    


from metropt.dtos import PredictionDTO

def test_prediction_roundtrip():
    dto = PredictionDTO(
        ts="2020-02-01T00:00:00+00:00", product_id="APU",
        fe_version="v1.0", model_version="20240101_000000",
        prob=0.73, label=1,
    )
    assert PredictionDTO.from_json(dto.to_json()) == dto

def test_prediction_prob_out_of_range_rejected():
    import pytest, pydantic
    with pytest.raises(pydantic.ValidationError):
        PredictionDTO(
            ts="2020-02-01T00:00:00+00:00", product_id="APU",
            fe_version="v1.0", model_version="v", prob=1.5, label=1,
        )


from metropt.dtos import Coordinates2DDTO

def test_coordinates_roundtrip():
    dto = Coordinates2DDTO(
        ts="2020-02-01T00:00:00+00:00", product_id="APU",
        dr_version="v1.0", x=2.14, y=-3.40,
    )
    assert Coordinates2DDTO.from_json(dto.to_json()) == dto