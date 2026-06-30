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