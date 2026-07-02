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


class FeatureVectorDTO(BaseModel):
    ts: str
    product_id: str
    fe_version: str            # which extractor produced this — versioning is mandatory
    features: Dict[str, float] # e.g. {"Motor_current_mean": 0.035, "TP3_std": 0.012, ...}

    @field_validator("ts")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "FeatureVectorDTO":
        return cls.model_validate_json(s)

    def to_ndarray(self) -> np.ndarray:
        # Sort keys for deterministic ordering — critical for ML.
        return np.array([self.features[k] for k in sorted(self.features)], dtype=float)



class PredictionDTO(BaseModel):
    ts: str
    product_id: str
    fe_version: str
    model_version: str
    prob: float
    label: int

    @field_validator("ts")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @field_validator("prob")
    @classmethod
    def _prob_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"prob must be in [0,1], got {v}")
        return v

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "PredictionDTO":
        return cls.model_validate_json(s)



class Coordinates2DDTO(BaseModel):
    ts: str
    product_id: str
    dr_version: str    # which reducer produced this
    x: float
    y: float

    @field_validator("ts")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "Coordinates2DDTO":
        return cls.model_validate_json(s)



class ShiftEventDTO(BaseModel):
    ts: str
    baseline_version: str
    stream: str                   # e.g. "features"
    score: float
    severity: str                 # "warn" | "alert" | "critical"
    window: Dict[str, Any]        # {size, start_ts, end_ts, top_features}

    @field_validator("ts")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @field_validator("severity")
    @classmethod
    def _sev(cls, v: str) -> str:
        if v not in {"warn", "alert", "critical"}:
            raise ValueError(f"severity must be warn|alert|critical, got {v}")
        return v

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "ShiftEventDTO":
        return cls.model_validate_json(s)



class ModelUpdateCompleteDTO(BaseModel):
    ts: str
    bundle: Dict[str, str]        # {"fe": ..., "primary": ..., "dr": ..., "dm": ...}
    eval_metrics: Dict[str, float]  # {"auroc": ..., "auprc": ..., "incumbent_auprc": ...}
    promotion_decision: str       # "promoted" | "rejected"

    @field_validator("ts")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @field_validator("promotion_decision")
    @classmethod
    def _pd(cls, v: str) -> str:
        if v not in {"promoted", "rejected"}:
            raise ValueError(f"promotion_decision must be promoted|rejected, got {v}")
        return v

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "ModelUpdateCompleteDTO":
        return cls.model_validate_json(s)