from __future__ import annotations
from datetime import datetime
from typing import Dict
import numpy as np
from pydantic import BaseModel, field_validator

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
