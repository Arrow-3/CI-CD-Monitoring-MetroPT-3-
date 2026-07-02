from __future__ import annotations
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from metropt.config import settings
from metropt.dtos import (
    ProcessedRawDataDTO, ShiftEventDTO, ModelUpdateCompleteDTO,
)
from metropt.services.base import BaseService
from metropt.services.feature_extractor import FeatureExtractorService
from metropt.storage import StorageManager


class TransferLearningService(BaseService):
    name = "transferlearningservice"
    input_topics = ["shift_events", "processed_raw_data"]
    output_topic = "model_update_complete"

    def __init__(self, *, connect: bool = True):
        super().__init__(connect=connect)
        self.sm = StorageManager()
        self.recent: Deque[ProcessedRawDataDTO] = deque(
            maxlen=settings.TL_RECENT_BUFFER_SIZE)
        self._last_retrain_ts: float = 0.0
        self.log.info("ready; buffer capacity=%d; trigger severities=%s",
                      settings.TL_RECENT_BUFFER_SIZE, settings.TL_TRIGGER_SEVERITY)

    def handle(self, message: str, topic: str | None = None) -> None:
        if topic == "processed_raw_data":
            self._on_processed(message)
        elif topic == "shift_events":
            self._on_shift(message)

    # ---- data ingestion ----

    def _on_processed(self, message: str) -> None:
        try:
            proc = ProcessedRawDataDTO.from_json(message)
        except Exception:
            self.log.exception("bad ProcessedRawDataDTO")
            return
        self.recent.append(proc)

    # ---- trigger + retrain ----

    def _on_shift(self, message: str) -> None:
        try:
            event = ShiftEventDTO.from_json(message)
        except Exception:
            self.log.exception("bad ShiftEventDTO")
            return
        if event.severity not in settings.TL_TRIGGER_SEVERITY:
            return

        now = time.time()
        if now - self._last_retrain_ts < settings.TL_RETRAIN_COOLDOWN_SECS:
            self.log.info("retrain cooldown active (%.0fs remaining), skipping",
                          settings.TL_RETRAIN_COOLDOWN_SECS - (now - self._last_retrain_ts))
            return

        if len(self.recent) < 1000:
            self.log.info("recent buffer too small (%d rows), skipping",
                          len(self.recent))
            return

        self._last_retrain_ts = now
        self.log.warning("shift '%s' score=%.3f — retraining on %d recent rows",
                         event.severity, event.score, len(self.recent))
        self._retrain_and_decide()

    def _retrain_and_decide(self) -> None:
        X, y = self._build_features_from_recent()
        pos = int(y.sum())
        self.log.info("candidate training set: %d rows, %d positives (%.2f%%)",
                      len(y), pos, 100 * pos / max(len(y), 1))
        if pos < settings.TL_MIN_POSITIVES:
            self.log.warning("too few positives (%d < %d) — skipping",
                             pos, settings.TL_MIN_POSITIVES)
            return

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42,
        )
        candidate = self._train(X_tr, y_tr)
        cand_metrics = self._eval(candidate, X_te, y_te)

        incumbent_auprc = self._eval_incumbent(X_te, y_te)
        promoted = cand_metrics["auprc"] >= (incumbent_auprc + settings.TL_PROMOTION_MARGIN)

        decision = "promoted" if promoted else "rejected"
        self.log.warning(
            "candidate auprc=%.3f incumbent=%.3f margin=%.3f → %s",
            cand_metrics["auprc"], incumbent_auprc,
            settings.TL_PROMOTION_MARGIN, decision.upper(),
        )

        # Save the candidate bundle unconditionally — even rejects are auditable.
        bundle_payload = {
            "model": candidate,
            "feature_names": list(X.columns),
            "metrics": {**cand_metrics, "incumbent_auprc": incumbent_auprc,
                        "decision": decision},
        }
        # Promotion = save under the same "primary" model_type (PrimaryModel picks up).
        # Rejection = save under "primary_rejected" for the record.
        model_type = settings.MODEL_TYPE if promoted else "primary_rejected"
        self.sm.save_model(settings.PROJECT, model_type, bundle_payload)
        new_version = self.sm.latest_model_version(settings.PROJECT, model_type)

        dr_v = self.sm.latest_model_version(settings.PROJECT, settings.DR_MODEL_TYPE) or "none"
        dm_v = self.sm.latest_model_version(settings.PROJECT, settings.DM_MODEL_TYPE) or "none"

        event = ModelUpdateCompleteDTO(
            ts=datetime.now(timezone.utc).isoformat(),
            bundle={
                "fe_version": settings.FE_VERSION,
                "primary_version": new_version,
                "dr_version": dr_v,
                "dm_version": dm_v,
            },
            eval_metrics={**cand_metrics, "incumbent_auprc": incumbent_auprc},
            promotion_decision=decision,
        )
        self.publish(self.output_topic, event)

    # ---- helpers ----

    def _build_features_from_recent(self) -> tuple[pd.DataFrame, np.ndarray]:
        fe = FeatureExtractorService(connect=False)
        feats, labels = [], []
        for proc in self.recent:
            fe.handle(proc.to_json())
            emitted = fe._last_published
            if emitted is None:
                continue
            feats.append(emitted.features)
            labels.append(proc.metadata.get("label", 0))
        X = pd.DataFrame(feats).sort_index(axis=1)
        y = np.array(labels, dtype=int)
        return X, y

    def _train(self, X_tr, y_tr) -> XGBClassifier:
        pos_weight = (len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1)
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=pos_weight, eval_metric="aucpr",
            tree_method="hist", random_state=42,
        )
        model.fit(X_tr, y_tr)
        return model

    def _eval(self, model, X_te, y_te) -> dict[str, float]:
        p = model.predict_proba(X_te)[:, 1]
        return {
            "auroc": float(roc_auc_score(y_te, p)),
            "auprc": float(average_precision_score(y_te, p)),
        }

    def _eval_incumbent(self, X_te, y_te) -> float:
        v = self.sm.latest_model_version(settings.PROJECT, settings.MODEL_TYPE)
        if v is None:
            return 0.0
        bundle = self.sm.load_model(settings.PROJECT, settings.MODEL_TYPE, v)
        incumbent = bundle["model"]
        # Align columns to incumbent's schema.
        cols = bundle["feature_names"]
        X_aligned = pd.DataFrame(
            {c: X_te[c] if c in X_te.columns else 0.0 for c in cols}
        )
        p = incumbent.predict_proba(X_aligned)[:, 1]
        return float(average_precision_score(y_te, p))


if __name__ == "__main__":
    TransferLearningService().run()