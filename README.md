# CI-CD-Monitoring-MetroPT-3-

## Running the DataGenerator under uv
```
uv sync                                         # creates .venv, installs everything 
docker compose up -d                            # Kafka + ZooKeeper
uv run python -m metropt.scripts.explore_data     # confirm column names first
uv run python -m metropt.scripts.create_topics    # the 7 topics
uv run python -m metropt.services.data_generator  # terminal 1: stream MetroPT-3
uv run python -m metropt.scripts.consume_raw      # terminal 2: verify RawDataDTO flowing   ```

# Adaptive Condition Monitoring on MetroPT-3

An event-driven ML system that continuously monitors industrial sensor data, detects distribution shifts, and adapts its models automatically — all without human intervention.

Built on the open-source [MetroPT-3 dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) (Metro do Porto air production unit, 15 sensors at 1 Hz), the system demonstrates a full **observe → detect → adapt** loop across seven Kafka-connected microservices, with a live Streamlit dashboard for operator supervision.

<p align="center">
  <!-- Replace with a wide screenshot of your dashboard, ideally captured during a drift event -->
  <img src="docs/images/dashboard_overview.png" alt="Live dashboard overview" width="900"/>
</p>

---

## Why this project

Most predictive-maintenance ML systems assume the world doesn't change. Real machines degrade; sensor distributions drift; the model trained on last quarter's data quietly stops working.

This system takes the opposite stance: **the world always drifts, and the ML pipeline should notice and respond.** When statistical drift crosses a threshold, a retraining service pulls the most recent labeled data, trains a candidate model, evaluates it against the incumbent, and either promotes or rejects it. If promoted, the serving model hot-reloads without a restart — from that message onward, predictions carry the new model version.

The architecture is deliberately modular: seven single-responsibility services, all communicating asynchronously over Kafka topics. Any service can be replaced, scaled, or rewritten independently.

---

## Architecture
┌─────────────────┐   raw_data_stream   ┌───────────────┐   processed_raw_data   ┌────────────────────┐
│  DataGenerator  │ ──────────────────> │   DataGate    │ ─────────────────────> │ FeatureExtractor   │
└─────────────────┘                     └───────────────┘                        └─────────┬──────────┘
│ feature_vectors
┌──────────────────────────────┼────────────────────────────┐
▼                              ▼                            ▼
┌───────────────┐             ┌────────────────┐            ┌────────────────────┐
│  DimReducer   │             │  PrimaryModel  │            │ DistributionMonitor│
└──────┬────────┘             └────────┬───────┘            └─────────┬──────────┘
│ coordinates_nd                │ predictions                  │ shift_events
▼                               ▼                              ▼
┌─────────────────────────────────────┐
│        TransferLearning             │◄─── processed_raw_data
└─────────────────┬───────────────────┘
│ model_update_complete
▼
(PrimaryModel hot-reloads)

| Service | Responsibility |
|---|---|
| **DataGenerator** | Streams the MetroPT-3 CSV at a configurable speed (default 600×). |
| **DataGate** | Validates schema, clips out-of-range values, flags quality issues, and derives labels from reported failure windows. |
| **FeatureExtractor** | Computes rolling-window statistics (mean/std/min/max over 30 s) for the 7 analog sensors + passes 8 digital signals through, producing 36-dim feature vectors. |
| **PrimaryModel** | XGBoost classifier serving live predictions with probability + label. Hot-reloads on promotion events. |
| **DimReducer** | PCA (with `StandardScaler`) fit on a bootstrap window; projects live feature vectors to 2D for visualization. |
| **DistributionMonitor** | Compares a rolling current window against a frozen baseline using per-feature KS distance. Emits `warn`/`alert`/`critical` events with cooldown to avoid alert storms. |
| **TransferLearning** | On `critical` events (with wall-clock cooldown), retrains an XGBoost candidate on recent labeled data, evaluates against the incumbent, and either promotes or rejects. |
| **UI (Streamlit)** | Live dashboard consuming all downstream topics for operator supervision. |
| **Orchestrator** | Single-command lifecycle for all services with dependency-ordered startup and clean shutdown. |

### Design principles

- **Event-driven.** No direct service-to-service calls. All communication is asynchronous over Kafka.
- **Contract-first.** Each message is a validated Pydantic DTO; contracts are versioned (`fe_version`, `model_version`, etc.) for auditability.
- **Bootstrap-then-serve.** Services that need calibration (DimReducer, DistributionMonitor) collect their first N samples live before beginning to publish.
- **Train-offline / serve-online.** PrimaryModel is trained via a standalone script; the service reads the latest artifact from `StorageManager`.
- **Adapt-in-place.** TransferLearning retrains during runtime; PrimaryModel picks up new models via `model_update_complete` messages without restart.
- **Every artifact is versioned.** UTC-timestamped model, baseline, and reducer files live under `data/artifacts/` for reproducibility.

---

## Live dashboard

<p align="center">
  <!-- Full dashboard shot: header metrics, scatter, probability timeline, drift score, tables -->
  <img src="docs/images/dashboard_full.png" alt="Full dashboard" width="900"/>
</p>

The dashboard consumes four Kafka topics and renders:

**Header metrics** — Live throughput (predictions/sec, coords/sec), buffered drift events, current severity pill (🟢 healthy · 🟡 warn · 🟠 alert · 🔴 critical), and the active model version.

<p align="center">
  <img src="docs/images/dashboard_header.png" alt="Header metrics" width="900"/>
</p>

**Feature-space scatter (PCA)** — 2D projection of the 36-dim feature vectors, colored by predicted failure probability. Healthy operation clusters in green; pre-failure regions drift toward red.

<p align="center">
  <img src="docs/images/dashboard_scatter.png" alt="PCA scatter" width="600"/>
</p>

**Failure probability over time** — The primary model's output, with the 0.5 decision threshold overlaid. Sudden climbs correspond to the model recognizing a degradation signature.

<p align="center">
  <img src="docs/images/dashboard_probability.png" alt="Probability timeline" width="600"/>
</p>

**Distribution drift score** — Per-message drift score with the warn/alert/critical thresholds drawn as horizontal reference lines. This is the diagnostic view of the DistributionMonitor.

<p align="center">
  <img src="docs/images/dashboard_drift.png" alt="Drift score timeline" width="900"/>
</p>

**Recent drift events & model-update history** — Tabular views of the last ten shift events (with top drifting features) and the last ten retraining decisions (with AUPRC vs incumbent).

<p align="center">
  <img src="docs/images/dashboard_tables.png" alt="Event tables" width="900"/>
</p>

---

## Quickstart

### Prerequisites
- Python 3.11
- [uv](https://github.com/astral-sh/uv) for dependency management
- Docker Desktop (for Kafka + ZooKeeper)
- MetroPT-3 CSV: download from [UCI](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) or [Kaggle](https://www.kaggle.com/datasets/nargesdavari/metropt3-air-compressor) and place at `data/MetroPT3(AirCompressor).csv`

### Setup

```bash
# 1. Install dependencies
uv sync

# 2. Start Kafka + ZooKeeper
docker compose up -d

# 3. Create the 7 Kafka topics
uv run python -m metropt.scripts.create_topics

# 4. Train the initial primary model (~30 s)
uv run python -m metropt.scripts.train_primary
```

### Run

```bash
# Terminal 1 — start the full pipeline
make run

# Terminal 2 — start the dashboard
make ui
```

The dashboard is served at [http://localhost:8501](http://localhost:8501).

Give the pipeline ~60 seconds after startup: FeatureExtractor bootstraps for 30 s of stream time, DimReducer collects 500 samples, DistributionMonitor collects 1000 baseline samples. Once these complete, the dashboard populates and drift detection becomes active.

### Stop

Ctrl-C the orchestrator terminal; it shuts services down in reverse dependency order. Kafka and ZooKeeper remain running until `docker compose down`.

---

## Repository layout

metropt/
├── config/settings.py          # Topics, thresholds, replay speed, failure windows
├── dtos.py                     # Pydantic contracts for all Kafka messages
├── kafka_utils.py              # Producer/consumer factories
├── storage.py                  # Versioned artifact persistence
├── services/
│   ├── base.py                 # Consume-loop base class
│   ├── data_generator.py       # → raw_data_stream
│   ├── data_gate.py            # raw_data_stream → processed_raw_data
│   ├── feature_extractor.py    # processed_raw_data → feature_vectors
│   ├── primary_model.py        # feature_vectors → predictions (+ hot-reload)
│   ├── dim_reducer.py          # feature_vectors → coordinates_nd
│   ├── distribution_monitor.py # feature_vectors → shift_events
│   └── transfer_learning.py    # shift_events + processed_raw_data → model_update_complete
├── scripts/
│   ├── create_topics.py
│   ├── train_primary.py
│   └── orchestrator.py
└── ui/dashboard.py             # Streamlit dashboard

---

## Testing

```bash
uv run pytest -v tests
```

The suite covers:

- **DTO round-trips** — every message contract serializes and deserializes cleanly.
- **DTO validators** — probability ranges, severity levels, promotion decisions are enforced at the boundary.
- **Service unit tests** — each service's `handle()` is exercised offline with mocked Kafka; input topic routing, bootstrap gating, cooldown behavior, and edge cases are covered.

Continuous integration runs the full suite plus lint on every PR via GitHub Actions (`.github/workflows/ci.yml`).

---

## Configuration reference

Key settings live in `metropt/config/settings.py`:

| Setting | Default | Purpose |
|---|---|---|
| `REPLAY_SPEED` | `600` | Stream acceleration factor (600× = 6 hours of data per minute of wall-clock) |
| `FE_WINDOW_SECONDS` | `30` | Rolling-window size for feature statistics |
| `DR_BOOTSTRAP_SIZE` | `500` | Samples before PCA is fit |
| `DM_BASELINE_SIZE` | `1000` | Samples in the drift baseline |
| `DM_WINDOW_SIZE` | `500` | Rolling current window for KS comparison |
| `DM_THRESHOLDS` | `{warn: 0.15, alert: 0.25, critical: 0.40}` | Drift severity tiers |
| `TL_TRIGGER_SEVERITY` | `{"critical"}` | Which severities cause retraining |
| `TL_RETRAIN_COOLDOWN_SECS` | `120` | Wall-clock minimum between retrains |
| `TL_PROMOTION_MARGIN` | `0.02` | AUPRC delta required to promote a candidate |

All environment-configurable via matching `metropt_*` env vars.

---

## References

- Veloso, B., Ribeiro, R. P., Gama, J. & Pereira, P. M. *The MetroPT dataset for predictive maintenance.* Scientific Data 9, 764 (2022).
- Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M. & Gama, J. *Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry.* IEEE DSAA (2021).

---

## License

MIT.