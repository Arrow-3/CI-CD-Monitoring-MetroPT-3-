from __future__ import annotations
from datetime import datetime
from typing import Dict
import numpy as np
from pydantic import BaseModel, field_validator
from typing import Any, List, Optional

class RawDataDTO(BaseModel):
    ts: str               # ISO-8601 timestamp
    product_id: str
    tool_id: str
    raw: Dict[str, float] # sensor name → value

    @field_validator("ts")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))  # raises if malformed
        return v

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "RawDataDTO":
        return cls.model_validate_json(s)

    def to_ndarray(self) -> np.ndarray:
        return np.array(list(self.raw.values()), dtype=float)



class ProcessedRawDataDTO(BaseModel):
    timestamp: str
    sensor_values: List[float]      # ordered list — order is fixed by sensor_names
    sensor_names: List[str]         # the schema, carried with the data
    metadata: Dict[str, Any]        # product_id, tool_id, label, etc.
    quality_flags: Dict[str, Any]   # missing_values, out_of_range, schema_version

    @field_validator("timestamp")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "ProcessedRawDataDTO":
        return cls.model_validate_json(s)

    def to_ndarray(self) -> np.ndarray:
        return np.array(self.sensor_values, dtype=float)