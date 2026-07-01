from __future__ import annotations
from collections import deque
from typing import Deque, Tuple
import numpy as np
import pandas as pd
from metropt.config import settings
from metropt.dtos import ProcessedRawDataDTO, FeatureVectorDTO
from metropt.services.base import BaseService


class FeatureExtractorService(BaseService):
    name = "featureextractorservice"
    input_topic = "processed_raw_data"
    output_topic = "feature_vectors"

    def __init__(self, *, connect: bool = True):
        super().__init__(connect=connect)
        self.window = pd.Timedelta(seconds=settings.FE_WINDOW_SECONDS)
        # One deque per analog sensor, each element is (timestamp, value).
        self.buffers: dict[str, Deque[Tuple[pd.Timestamp, float]]] = {
            s: deque() for s in settings.FE_ANALOG_FEATURES
        }
        self._bootstrap_done = False
        self._first_ts: pd.Timestamp | None = None
        self.log.info(
            "window=%ds, %d analog sensors × %d stats = %d rolling features",
            settings.FE_WINDOW_SECONDS, len(settings.FE_ANALOG_FEATURES),
            len(settings.FE_STATS),
            len(settings.FE_ANALOG_FEATURES) * len(settings.FE_STATS),
        )

    def handle(self, message: str) -> None:
        try:
            proc = ProcessedRawDataDTO.from_json(message)
        except Exception:
            self.log.exception("rejecting malformed ProcessedRawDataDTO")
            return

        ts = pd.Timestamp(proc.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        values = dict(zip(proc.sensor_names, proc.sensor_values))
        self._update_buffers(ts, values)

        if not self._window_full(ts):
            return  # bootstrap — not enough history yet

        if not self._bootstrap_done:
            self.log.info("bootstrap complete; emitting features from %s", ts)
            self._bootstrap_done = True

        features = self._compute_features(values)
        out = FeatureVectorDTO(
            ts=proc.timestamp,
            product_id=proc.metadata.get("product_id", "unknown"),
            fe_version=settings.FE_VERSION,
            features=features,
        )
        self.publish(self.output_topic, out)

    # ---- internals ----

    def _update_buffers(self, ts: pd.Timestamp, values: dict[str, float]) -> None:
        if self._first_ts is None:
            self._first_ts = ts
        cutoff = ts - self.window
        for sensor, buf in self.buffers.items():
            if sensor in values:
                buf.append((ts, values[sensor]))
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def _window_full(self, ts: pd.Timestamp) -> bool:
        return self._first_ts is not None and (ts - self._first_ts) >= self.window

    def _compute_features(self, instant: dict[str, float]) -> dict[str, float]:
        feats: dict[str, float] = {}

        # Rolling stats per analog sensor.
        for sensor in settings.FE_ANALOG_FEATURES:
            buf = self.buffers[sensor]
            if not buf:
                # Sensor missing from the stream — fill with safe zeros.
                for stat in settings.FE_STATS:
                    feats[f"{sensor}_{stat}"] = 0.0
                continue
            arr = np.fromiter((v for _, v in buf), dtype=float)
            feats[f"{sensor}_mean"] = float(arr.mean())
            feats[f"{sensor}_std"]  = float(arr.std()) if arr.size > 1 else 0.0
            feats[f"{sensor}_min"]  = float(arr.min())
            feats[f"{sensor}_max"]  = float(arr.max())

        # Digital signals: pass through instantaneous value.
        for sensor in settings.DIGITAL_COLS:
            if sensor in instant:
                feats[sensor] = float(instant[sensor])

        return feats


if __name__ == "__main__":
    FeatureExtractorService().run()