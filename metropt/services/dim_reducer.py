from __future__ import annotations
from typing import List
import numpy as np
from sklearn.decomposition import PCA
from metropt.config import settings
from metropt.dtos import FeatureVectorDTO, Coordinates2DDTO
from metropt.services.base import BaseService
from metropt.storage import StorageManager
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class DimReducerService(BaseService):
    name = "dimreducerservice"
    input_topic = "feature_vectors"
    output_topic = "coordinates_nd"

    def __init__(self, *, connect: bool = True):
        super().__init__(connect=connect)
        self.sm = StorageManager()
        self.reducer: PCA | None = None
        self.feature_names: List[str] = []
        self.dr_version: str = "none"
        self.buffer: list[dict[str, float]] = []
        self._try_load_existing()

    def _try_load_existing(self) -> None:
        v = self.sm.latest_model_version(settings.PROJECT, settings.DR_MODEL_TYPE)
        if v is None:
            self.log.info("no reducer found; will bootstrap on first %d samples",
                          settings.DR_BOOTSTRAP_SIZE)
            return
        bundle = self.sm.load_model(settings.PROJECT, settings.DR_MODEL_TYPE, v)
        self.reducer = bundle["reducer"]
        self.feature_names = bundle["feature_names"]
        self.dr_version = v
        self.log.info("loaded reducer %s (%d input features → 2D)",
                      v, len(self.feature_names))

    def handle(self, message: str) -> None:
        try:
            fv = FeatureVectorDTO.from_json(message)
        except Exception:
            self.log.exception("rejecting malformed FeatureVectorDTO")
            return

        # Bootstrap phase
        if self.reducer is None:
            self.buffer.append(fv.features)
            if len(self.buffer) < settings.DR_BOOTSTRAP_SIZE:
                if len(self.buffer) % 100 == 0:
                    self.log.info("bootstrap: %d / %d",
                                  len(self.buffer), settings.DR_BOOTSTRAP_SIZE)
                return
            self._fit_from_buffer()
            # fall through to also transform this last bootstrap sample

        # Steady-state phase
        x = np.array([fv.features.get(n, 0.0) for n in self.feature_names],
                     dtype=float).reshape(1, -1)
        xy = self.reducer.transform(x)[0]
        coord = Coordinates2DDTO(
            ts=fv.ts, product_id=fv.product_id,
            dr_version=self.dr_version,
            x=float(xy[0]), y=float(xy[1]),
        )
        self.publish(self.output_topic, coord)

    def _fit_from_buffer(self) -> None:
        self.feature_names = sorted(self.buffer[0].keys())
        X = np.array(
            [[row.get(n, 0.0) for n in self.feature_names] for row in self.buffer],
            dtype=float,
        )
        reducer = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=settings.DR_N_COMPONENTS, random_state=42)),
        ])
        reducer.fit(X)
        self.reducer = reducer

        pca = reducer.named_steps["pca"]
        var = float(pca.explained_variance_ratio_.sum())
        self.log.info("fit PCA on %d×%d (standardized); explained variance = %.1f%%",
                      *X.shape, var * 100)

        bundle = {
            "reducer": reducer,
            "feature_names": self.feature_names,
            "n_bootstrap": len(self.buffer),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
        path = self.sm.save_model(settings.PROJECT, settings.DR_MODEL_TYPE, bundle)
        self.dr_version = self.sm.latest_model_version(
            settings.PROJECT, settings.DR_MODEL_TYPE)
        self.log.info("saved reducer → %s", path)
        self.buffer.clear()  # release memory



if __name__ == "__main__":
    DimReducerService().run()