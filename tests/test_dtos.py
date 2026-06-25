from metropt.dtos import RawDataDTO

def test_rawdata_roundtrip():
    dto = RawDataDTO(ts="2020-04-18T00:00:00+00:00", product_id="APU",
                     tool_id="metro-1", raw={"TP2": 1.0, "Motor_current": 3.4})
    again = RawDataDTO.from_json(dto.to_json())
    assert again == dto
    assert again.to_ndarray().tolist() == [1.0, 3.4]
