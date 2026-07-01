from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import joblib

DATA_ROOT = Path("data/artifacts")


class StorageManager:
    """Versioned local persistence for models, training data, and state.

    Layout:
      data/artifacts/models/{project}/{model_type}/model_{ts}.joblib
      data/artifacts/training/{project}/{name}_{ts}.joblib
      data/artifacts/state/{name}.json
    """

    def __init__(self, root: Path = DATA_ROOT):
        self.root = root

    # ---- models ----
    def save_model(self, project: str, model_type: str, model: Any,
                   version: str | None = None) -> Path:
        version = version or self._timestamp()
        path = self.root / "models" / project / model_type / f"model_{version}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return path

    def load_model(self, project: str, model_type: str, version: str) -> Any:
        return joblib.load(self.root / "models" / project / model_type
                           / f"model_{version}.joblib")

    def latest_model_version(self, project: str, model_type: str) -> str | None:
        d = self.root / "models" / project / model_type
        if not d.exists():
            return None
        files = sorted(d.glob("model_*.joblib"))
        return files[-1].stem.removeprefix("model_") if files else None

    # ---- training data ----
    def save_training_data(self, project: str, name: str, data: Any,
                           version: str | None = None) -> Path:
        version = version or self._timestamp()
        path = self.root / "training" / project / f"{name}_{version}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)
        return path

    # ---- state ----
    def save_state(self, name: str, state: Any) -> Path:
        path = self.root / "state" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(state, indent=2, default=str))
        except TypeError:
            path = path.with_suffix(".joblib")
            joblib.dump(state, path)
        return path

    def load_state(self, name: str) -> Any:
        p_json = self.root / "state" / f"{name}.json"
        p_jl   = self.root / "state" / f"{name}.joblib"
        if p_json.exists():
            return json.loads(p_json.read_text())
        if p_jl.exists():
            return joblib.load(p_jl)
        raise FileNotFoundError(f"no state named {name}")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")