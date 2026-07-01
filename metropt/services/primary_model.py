from __future__ import annotations
import numpy as np
from metropt.config import settings
from metropt.dtos import FeatureVectorDTO, PredictionDTO
from metropt.services.base import BaseService
from metropt.storage import StorageManager


class PrimaryModelService(BaseService):
    name = "primarymodelservice"
    input_topic = "feature_vectors"
    output_topic = "predictions"

    def __init__(self, *, connect: bool = True):
        super().__init__(connect=connect)
        self.sm = StorageManager()
        self.model = None
        self.feature_names: list[str] = []
        self.model_version: str = "none"
        self._load_latest_model()

    def _load_latest_model(self) -> None:
        v = self.sm.latest_model_version(settings.PROJECT, settings.MODEL_TYPE)
        if v is None:
            self.log.warning("no trained model found — run train_primary first")
            return
        bundle = self.sm.load_model(settings.PROJECT, settings.MODEL_TYPE, v)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.model_version = v
        self.log.info("loaded model %s with %d features", v, len(self.feature_names))

    def handle(self, message: str) -> None:
        if self.model is None:
            return
        try:
            fv = FeatureVectorDTO.from_json(message)
        except Exception:
            self.log.exception("rejecting malformed FeatureVectorDTO")
            return

        # Align features to the training schema. Missing → 0; extra → ignored.
        x = np.array([fv.features.get(n, 0.0) for n in self.feature_names],
                     dtype=float).reshape(1, -1)
        prob = float(self.model.predict_proba(x)[0, 1])
        label = int(prob >= settings.MODEL_PRED_THRESHOLD)

        pred = PredictionDTO(
            ts=fv.ts, product_id=fv.product_id, fe_version=fv.fe_version,
            model_version=self.model_version, prob=prob, label=label,
        )
        self.publish(self.output_topic, pred)


if __name__ == "__main__":
    PrimaryModelService().run()