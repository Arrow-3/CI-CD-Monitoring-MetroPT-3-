from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from metropt.config import settings
from metropt.services.data_gate import DataGateService
from metropt.services.feature_extractor import FeatureExtractorService
from metropt.dtos import RawDataDTO, ProcessedRawDataDTO, FeatureVectorDTO
from metropt.storage import StorageManager


def _build_training_set() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(settings.DATA_PATH)
    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=df.columns[0])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) > settings.TRAIN_SAMPLE_LIMIT:
        df = df.iloc[:: max(1, len(df) // settings.TRAIN_SAMPLE_LIMIT)].copy()
        print(f"downsampled to {len(df)} rows")

    gate = DataGateService(connect=False)
    fe   = FeatureExtractorService(connect=False)

    cols = [c for c in settings.ANALOG_COLS + settings.DIGITAL_COLS
            if c in df.columns]
    feats, labels = [], []

    for _, row in df.iterrows():
        raw = RawDataDTO(
            ts=row["timestamp"].isoformat(), product_id="APU", tool_id="metro-1",
            raw={c: float(row[c]) for c in cols},
        ).to_json()

        gate.handle(raw)
        processed = gate._last_published
        if processed is None:
            continue
        fe.handle(processed.to_json())
        emitted = fe._last_published
        if emitted is None:
            continue
        feats.append(emitted.features)
        labels.append(processed.metadata["label"])

    X = pd.DataFrame(feats).sort_index(axis=1)
    y = np.array(labels, dtype=int)
    return X, y


def main():
    print("building features from CSV...")
    X, y = _build_training_set()
    print(f"features: {X.shape}; positives: {int(y.sum())} ({y.mean()*100:.2f}%)")
    if y.sum() == 0:
        raise SystemExit("no positive labels — check FAILURE_WINDOWS")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=settings.TRAIN_TEST_SPLIT,
        stratify=y, random_state=42,
    )
    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    print(f"scale_pos_weight: {pos_weight:.1f}")

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        tree_method="hist", random_state=42,
    )
    model.fit(X_tr, y_tr)

    p_te = model.predict_proba(X_te)[:, 1]
    metrics = {
        "auroc": float(roc_auc_score(y_te, p_te)),
        "auprc": float(average_precision_score(y_te, p_te)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "feature_names": list(X.columns),
    }
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("feature_names",)}, indent=2))

    sm = StorageManager()
    bundle = {"model": model, "feature_names": list(X.columns), "metrics": metrics}
    path = sm.save_model(settings.PROJECT, settings.MODEL_TYPE, bundle)
    print(f"saved → {path}")


if __name__ == "__main__":
    main()