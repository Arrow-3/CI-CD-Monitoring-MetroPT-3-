''' DataGateService is the validation and enrichment step between raw ingestion and downstream ML/processing. It sits in the Kafka pipeline as:

raw_data_stream → DataGateService → processed_raw_data.

Publishes a ProcessedRawDataDTO to processed_raw_data with:

- Ordered sensor_values + sensor_names
- Metadata (product_id, tool_id, label)
- Quality flags (missing_values, out_of_range, schema_version '''



from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import math
import pandas as pd
from metropt.config import settings
from metropt.dtos import RawDataDTO, ProcessedRawDataDTO
from metropt.services.base import BaseService


class DataGateService(BaseService):
    name = "datagateservice"
    input_topic = "raw_data_stream"
    output_topic = "processed_raw_data"

    def __init__(self):
        super().__init__()
        # Pre-parse failure windows once, as timezone-aware UTC.
        self._windows = [
            (pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC"))
            for s, e in settings.FAILURE_WINDOWS
        ]
        self._pre = pd.Timedelta(hours=settings.PRE_FAILURE_HOURS)
        self.log.info(
            "loaded %d failure windows; pre-failure horizon=%dh",
            len(self._windows), settings.PRE_FAILURE_HOURS,
        )

    def handle(self, message: str) -> None:
        try:
            raw = RawDataDTO.from_json(message)
        except Exception:
            self.log.exception("rejecting malformed RawDataDTO")
            return

        values, names, flags = self._validate_and_normalize(raw.raw)
        label = self._label_for(raw.ts)

        processed = ProcessedRawDataDTO(
            timestamp=raw.ts,
            sensor_values=values,
            sensor_names=names,
            metadata={
                "product_id": raw.product_id,
                "tool_id": raw.tool_id,
                "label": label,
            },
            quality_flags={**flags, "schema_version": settings.SCHEMA_VERSION},
        )
        self.publish(self.output_topic, processed)

    # ---- internals ----

    def _validate_and_normalize(
        self, raw: dict[str, float]
    ) -> tuple[list[float], list[str], dict[str, Any]]:
        names = sorted(raw.keys())  # deterministic ordering
        values: list[float] = []
        missing: list[str] = []
        out_of_range: list[str] = []

        for n in names:
            v = raw[n]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                missing.append(n)
                v = 0.0  # safe default; flagged for downstream awareness
            elif n in settings.SENSOR_RANGES:
                lo, hi = settings.SENSOR_RANGES[n]
                if not (lo <= v <= hi):
                    out_of_range.append(n)
                    v = max(lo, min(hi, v))  # clip analog sensors to configured min/max range
            elif n in settings.DIGITAL_COLS:
                if v not in settings.DIGITAL_VALID:
                    out_of_range.append(n)
                    v = 1.0 if v >= 0.5 else 0.0 # Snaps invalid digital values to 0.0 or 1.0
            values.append(float(v))

        return values, names, {
            "missing_values": missing or False,
            "out_of_range": out_of_range or False,
        }

    def _label_for(self, ts: str) -> int:
        t = pd.Timestamp(ts) 
        if t.tzinfo is None:    # Uses known failure windows from settings
            t = t.tz_localize("UTC")
        for start, end in self._windows:
            if (start - self._pre) <= t <= end: # Marks label=1 if the timestamp falls within a failure window or the pre-failure horizon (PRE_FAILURE_HOURS, default 6h before)
                return 1
        return 0


if __name__ == "__main__":
    DataGateService().run()