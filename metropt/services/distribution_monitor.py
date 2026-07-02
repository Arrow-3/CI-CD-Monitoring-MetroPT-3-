from __future__ import annotations
from collections import deque
from typing import Deque, Dict, List
import numpy as np
from scipy import stats
from metropt.config import settings
from metropt.dtos import FeatureVectorDTO, ShiftEventDTO
from metropt.services.base import BaseService
from metropt.storage import StorageManager


class DistributionMonitorService(BaseService):
    name = "distributionmonitorservice"
    input_topic = "feature_vectors"
    output_topic = "shift_events"

    def __init__(self, *, connect: bool = True):
        super().__init__(connect=connect)
        self.sm = StorageManager()

        self.baseline: np.ndarray | None = None      # (N, F)
        self.feature_names: List[str] = []
        self.baseline_version: str = "none"

        self.buffer: list[dict[str, float]] = []     # for baseline bootstrap
        self.window: Deque[np.ndarray] = deque(maxlen=settings.DM_WINDOW_SIZE)
        self.window_ts: Deque[str] = deque(maxlen=settings.DM_WINDOW_SIZE)

        self._cooldown: Dict[str, int] = {"warn": 0, "alert": 0, "critical": 0}
        self._try_load_baseline()

    def _try_load_baseline(self) -> None:
        v = self.sm.latest_model_version(settings.PROJECT, settings.DM_MODEL_TYPE)
        if v is None:
            self.log.info("no baseline found; will collect first %d samples",
                          settings.DM_BASELINE_SIZE)
            return
        bundle = self.sm.load_model(settings.PROJECT, settings.DM_MODEL_TYPE, v)
        self.baseline = bundle["baseline"]
        self.feature_names = bundle["feature_names"]
        self.baseline_version = v
        self.log.info("loaded baseline %s (%d samples × %d features)",
                      v, self.baseline.shape[0], self.baseline.shape[1])

    def handle(self, message: str) -> None:
        try:
            fv = FeatureVectorDTO.from_json(message)
        except Exception:
            self.log.exception("rejecting malformed FeatureVectorDTO")
            return

        # Baseline bootstrap
        if self.baseline is None:
            self.buffer.append(fv.features)
            if len(self.buffer) % 200 == 0:
                self.log.info("baseline: %d / %d",
                              len(self.buffer), settings.DM_BASELINE_SIZE)
            if len(self.buffer) < settings.DM_BASELINE_SIZE:
                return
            self._freeze_baseline()

        # Steady state — vectorize this row into the current window
        x = np.array([fv.features.get(n, 0.0) for n in self.feature_names],
                     dtype=float)
        self.window.append(x)
        self.window_ts.append(fv.ts)

        # Cool down all counters
        for k in self._cooldown:
            self._cooldown[k] = max(0, self._cooldown[k] - 1)

        if len(self.window) < settings.DM_WINDOW_SIZE:
            return  # not enough current data yet

        score, top = self._compute_drift()
        severity = self._severity(score)
        if severity is None:
            return
        if self._cooldown[severity] > 0:
            return

        self._cooldown[severity] = settings.DM_COOLDOWN_MESSAGES

        event = ShiftEventDTO(
            ts=fv.ts,
            baseline_version=self.baseline_version,
            stream="features",
            score=float(score),
            severity=severity,
            window={
                "size": len(self.window),
                "start_ts": self.window_ts[0],
                "end_ts": self.window_ts[-1],
                "top_features": top,
            },
        )
        self.log.warning("SHIFT %s score=%.3f top=%s",
                         severity, score, [f["name"] for f in top])
        self.publish(self.output_topic, event)

    # ---- internals ----

    def _freeze_baseline(self) -> None:
        self.feature_names = sorted(self.buffer[0].keys())
        self.baseline = np.array(
            [[row.get(n, 0.0) for n in self.feature_names] for row in self.buffer],
            dtype=float,
        )
        bundle = {
            "baseline": self.baseline,
            "feature_names": self.feature_names,
            "n_samples": len(self.buffer),
        }
        self.sm.save_model(settings.PROJECT, settings.DM_MODEL_TYPE, bundle)
        self.baseline_version = self.sm.latest_model_version(
            settings.PROJECT, settings.DM_MODEL_TYPE)
        self.log.info("froze baseline %s (%d × %d); saved.",
                      self.baseline_version, *self.baseline.shape)
        self.buffer.clear()

    def _compute_drift(self) -> tuple[float, list[dict]]:
        current = np.array(self.window)             # (W, F)
        per_feature: list[tuple[str, float]] = []
        for i, name in enumerate(self.feature_names):
            b = self.baseline[:, i]
            c = current[:, i]
            if b.std() == 0 and c.std() == 0:
                d = 0.0                              # both constant, no signal
            else:
                d = float(stats.ks_2samp(b, c).statistic)
            per_feature.append((name, d))

        per_feature.sort(key=lambda p: p[1], reverse=True)
        topk = per_feature[: settings.DM_TOP_K]
        score = float(np.mean([d for _, d in topk]))
        return score, [{"name": n, "ks": round(d, 4)} for n, d in topk]

    def _severity(self, score: float) -> str | None:
        t = settings.DM_THRESHOLDS
        if score >= t["critical"]: return "critical"
        if score >= t["alert"]:    return "alert"
        if score >= t["warn"]:     return "warn"
        return None


if __name__ == "__main__":
    DistributionMonitorService().run()