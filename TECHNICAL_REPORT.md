1. Executive summary
This project is a modular, event-driven machine learning system that continuously monitors industrial sensor data, detects distribution shifts in real time, and adapts its serving model automatically — without human intervention and without service restart. It is built on the open-source MetroPT-3 dataset (Air Production Unit of a Portuguese metro train, 15 sensors sampled at 1 Hz, February–August 2020) and demonstrates a complete observe → detect → adapt loop across seven asynchronously coupled microservices.
The core technical claim is that ML pipelines for physical systems must treat distribution drift as a first-class runtime concern rather than a training-time assumption. Concretely, this means: (a) every service holds a small, well-defined state and communicates only through versioned message contracts; (b) a dedicated monitor continuously compares live feature distributions to a frozen baseline; and (c) an in-process retraining service is triggered by drift, evaluates a candidate against the incumbent on the most recent labeled data, and hot-swaps the serving model if the candidate improves.
Headline properties:
•	7 services + 7 Kafka topics, orchestrated single-command via a lightweight Python supervisor.
•	Offline-trained XGBoost primary model with AUROC 0.985 / AUPRC 0.899 on stratified evaluation.
•	KS-based distribution monitor with three severity tiers and cooldown hysteresis.
•	Runtime model promotion with configurable AUPRC margin.
•	Two operator interfaces: a Streamlit dashboard for quick iteration and a production-style React + FastAPI + Server-Sent Events dashboard for demonstration.
•	Full test suite at three layers (DTO contracts, service handlers, end-to-end integration) with GitHub Actions CI.
________________________________________
2. Problem statement and motivation
2.1 The failure mode of static ML deployments
Machine learning models trained on historical data assume the future looks like the past. For physical industrial systems, this assumption is systematically wrong. Compressors wear, seals degrade, ambient conditions shift with the seasons, and maintenance interventions reset the sensor signatures the model has learned. A model trained at time t₀ is quietly wrong by t₀ + Δ — not catastrophically, not visibly, but by enough that its precision and recall drift and its predictions become progressively less trustworthy.
The typical response is periodic retraining on a schedule (weekly, monthly, quarterly). This has two well-known deficiencies: schedules are decoupled from the actual rate of change in the data, and periodic retraining cannot respond to sudden regime shifts (a component replacement, a workload change, a sensor recalibration).
This project takes the opposite position: the system itself should notice when the world has changed and decide whether to update. The pipeline runs in perpetuity; drift is monitored continuously; retraining is event-triggered by evidence of distribution shift, not calendar time.
2.2 What "adaptive" concretely requires
Making adaptation runtime-safe requires a specific set of architectural properties:
•	Versioned contracts. Every message includes the version of every artifact that produced it (feature extractor, model, dimensionality reducer, drift baseline). When TransferLearning promotes a new model, downstream consumers can immediately answer "which model produced this prediction?"
•	Stateless-per-message services. Each service consumes a message, does one job, publishes the result. No shared session state. This makes the retraining decision safe: TransferLearning can save a new model to disk and PrimaryModel picks it up on the next model_update_complete event — no restart, no coordination.
•	Bounded memory everywhere. Rolling-window computations use deques with maxlen. Baselines are frozen once and persisted. The dashboard buffers cap at 2000 messages. There is no unbounded growth anywhere in the runtime.
•	Auditability. Every promotion decision is emitted as a model_update_complete DTO with the candidate metrics, the incumbent's score on the same evaluation set, and the promotion verdict. Rejected candidates are still persisted for post-hoc analysis.
•	Graceful degradation. No trained model? The serving service simply drops incoming feature vectors silently. No baseline? The drift monitor buffers samples until the baseline is frozen. Cold-start scenarios never crash the pipeline.
2.3 Why event-driven microservices for this specific problem
Three properties of the problem justify the architecture:
•	Different services have different rhythms. DataGenerator produces at 1 Hz (or replay-accelerated). FeatureExtractor is stateful (30-second rolling windows). PrimaryModel is stateless inference (microseconds per row). TransferLearning is a heavy batch job (seconds to minutes) triggered rarely. Coupling these synchronously would make the fastest run at the pace of the slowest.
•	Fan-out is natural. Feature vectors go to three consumers simultaneously (PrimaryModel, DimReducer, DistributionMonitor). Kafka topics fan out by design; direct calls would require the producer to know all consumers.
•	The retraining loop is fundamentally asynchronous. Drift detection happens on one stream, model retraining reads a different one, model promotion writes to a third. Trying to express this with function calls yields a distributed state machine masquerading as sequential code. Kafka topics make the state machine explicit.
________________________________________
3. Dataset — MetroPT-3
3.1 Source and scale
MetroPT-3 is a publicly available dataset released by the Metro do Porto (Portugal) via UCI Machine Learning Repository. It contains ~1.5 million rows sampled at 1 Hz between February and August 2020, from sensors installed on the Air Production Unit (APU) of a metro train. The APU is a rooftop compressor system feeding the pneumatic secondary suspension — an unglamorous but safety-relevant subsystem whose failure removes the vehicle from service.
The dataset comprises 15 sensor channels:
Type	Channels
Analog (7)	TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current
Digital (8)	COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses
Analog channels are pressures (bar), temperatures (°C), and current (A). Digital channels are on/off states of valves, switches, and control signals.
3.2 Failure inventory
Four failure events are documented during the collection period:
Failure	Start	End	Reported cause
1	2020-04-18 00:00	2020-04-18 23:59	Air leak
2	2020-05-29 23:30	2020-05-30 06:00	Air leak
3	2020-06-05 10:00	2020-06-07 14:30	Air leak
4	2020-07-15 14:30	2020-07-15 19:00	Air leak
Four events over 6+ months of data means the positive class is intrinsically rare (~2.5% of rows after applying a 6-hour pre-failure window as the labeled positive region). Any modeling strategy has to account for this severe imbalance, and any evaluation strategy that uses random train/test splits will overstate performance.
3.3 Label derivation
The dataset ships unlabeled. Labels are derived at ingestion time by DataGate: for each row, if its timestamp falls within PRE_FAILURE_HOURS before or during any documented failure window, the row is labeled positive (1), otherwise negative (0). The pre-failure horizon is 6 hours by default — a compromise between two failure modes:
•	Too short (e.g., 30 minutes): the positive class is even rarer, the model has almost nothing to learn from, and predictions cannot give useful lead time to operators.
•	Too long (e.g., 24 hours): labels include long periods of "normal" operation, diluting the failure signal and encouraging the model to memorize temporal patterns rather than degradation signatures.
This is a deliberate design choice with a known cost: models trained with any labeling of pre-failure windows will inflate their evaluation metrics if the split is random (because features shortly before failure share rolling-window history with features during failure). A rigorous evaluation would split by failure event rather than by row; this project uses random splits for computational efficiency and calls out the limitation explicitly (§13).
________________________________________
4. Architectural overview
4.1 Topology
raw_data_streamprocessed_raw_dataprocessed_raw_datafeature_vectorsfeature_vectorsfeature_vectorspredictionscoordinates_ndshift_eventsshift_eventsmodel_update_completemodel_update_completeSSESSEDataGeneratorDataGateFeatureExtractorTransferLearningPrimaryModelDimReducerDistributionMonitorSSE BridgeReact DashboardStreamlit Dashboard
Seven backend services (blue = producer, pink = ML, yellow = retrainer) exchange messages via seven Kafka topics. The SSE bridge (purple) translates Kafka to browser-friendly HTTP streams. The React and Streamlit dashboards are consumers of the bridge.
4.2 Architectural principles
Six principles govern the design; each embeds a trade-off worth naming explicitly.
Event-driven over synchronous. Services never call each other directly. Every interaction is mediated by a Kafka topic. This costs latency (typically 1–5 ms per hop) but purchases decoupling: a slow service backs up its own topic without stalling upstream producers, and swapping any service's implementation requires no changes to its neighbors.
Contract-first via versioned DTOs. Each message is a Pydantic v2 DTO with field-level validators. Contracts are versioned (fe_version, model_version, dr_version, dm_version) so that consumers can reason about which upstream artifact produced any given message. This costs a small amount of per-message validation overhead and forces up-front schema discipline; it purchases forward compatibility and auditability.
Bootstrap-then-serve for stateful services. DimReducer, DistributionMonitor, and (in a different sense) FeatureExtractor all require calibration before they can produce meaningful output. Each collects its first N samples silently, fits its state, persists it, and only then begins publishing. On restart, saved state is loaded from disk and bootstrap is skipped. This costs a startup window of silence (typically 30–60 seconds); it purchases correctness — services do not emit garbage during warmup.
Train-offline / serve-online with hot-reload. The initial PrimaryModel is trained by a standalone Python script (scripts/train_primary.py) using the same feature-extraction code the live service will run. This costs a manual bootstrap step; it purchases a clean separation between the batch concerns of training (large data, expensive fits, evaluation) and the streaming concerns of serving (single-row inference at microsecond scale).
Adapt-in-place via model_update_complete. TransferLearning writes new model artifacts to disk and emits a model_update_complete event. PrimaryModel subscribes to this event and, on promoted verdicts, reloads the model artifact atomically — no restart, no service interruption. This costs a small amount of dual-subscription complexity in PrimaryModel; it purchases zero-downtime model updates.
Every artifact versioned via StorageManager. Models, drift baselines, and dimensionality reducers all persist through a single API that stamps each artifact with a UTC timestamp. Loading always resolves to the latest version by default; explicit versions are available for rollback. This costs a small disk footprint; it purchases reproducibility — any historical decision can be re-run against the exact artifacts it used.
________________________________________
5. Kafka topics and DTOs
5.1 Topic inventory
The 7 topics form a directed acyclic graph of message flow. Each topic uses 3 partitions and 1 replica (single-broker Docker Compose Kafka).
Topic	Producer	Consumers	Payload
raw_data_stream	DataGenerator	DataGate	Sensor readings + metadata
processed_raw_data	DataGate	FeatureExtractor, TransferLearning	Validated + labeled rows
feature_vectors	FeatureExtractor	PrimaryModel, DimReducer, DistributionMonitor	36-dim feature vectors
predictions	PrimaryModel	SSE Bridge, dashboards	Failure probability + label
coordinates_nd	DimReducer	SSE Bridge, dashboards	2D PCA projections
shift_events	DistributionMonitor	TransferLearning, SSE Bridge	Drift severity + top features
model_update_complete	TransferLearning	PrimaryModel, SSE Bridge	Promotion verdict + metrics
5.2 DTO summary
All DTOs use Pydantic v2 with field-level validators. Timestamps are ISO-8601. Enumerated fields (severity, promotion_decision) use Literal types.
DTO	Key fields	Validators
RawDataDTO	ts, product_id, tool_id, raw{}	Sensor values must be numeric
ProcessedRawDataDTO	ts, sensor_values[], sensor_names[], metadata{label}, quality_flags	Label ∈ {0,1}; flags populated
FeatureVectorDTO	ts, product_id, fe_version, features{}	Non-empty features dict
PredictionDTO	ts, model_version, prob, label	0 ≤ prob ≤ 1; label ∈ {0,1}
Coordinates2DDTO	ts, dr_version, x, y	x, y finite floats
ShiftEventDTO	ts, baseline_version, stream, score, severity, window	severity ∈ {warn, alert, critical}
ModelUpdateCompleteDTO	ts, bundle, eval_metrics, promotion_decision	promotion_decision ∈ {promoted, rejected}
Every message in the entire system belongs to exactly one of these seven types, and every service produces or consumes only well-typed instances. Schema violations fail loudly at the boundary, never silently corrupting downstream state.
5.3 Message flow for a drift-triggered retrain
Kafka topicsPrimaryModelTransferLearningDistributionMonitorFeatureExtractorDataGateDataGeneratorKafka topicsPrimaryModelTransferLearningDistributionMonitorFeatureExtractorDataGateDataGeneratorSteady state (drift-free)Drift threshold crossedRetrain gates:severity=critical? cooldown expired?positives ≥ threshold?Subsequent predictions carry new model_versionraw_data_stream (row)1raw2processed_raw_data (labeled row)3processed row4processed row (buffered)5feature_vectors6features7features8predictions9shift_events (severity=critical)10shift trigger11Train candidate on recent buffer12Evaluate candidate vs incumbent13model_update_complete (promoted)14reload signal15Load new artifact from disk16
The critical detail is that TransferLearning's promotion is not an immediate action — it is a message published to a topic. PrimaryModel's hot-reload happens on its next message from model_update_complete, which arrives on its own consumer group's schedule. The two services are never directly coupled; the topic is the coordination mechanism.
________________________________________
6. The seven services
Each subsection follows the same structure: responsibility, contracts, state and lifecycle, key algorithm, configuration, and honest trade-offs.
6.1 DataGenerator
Responsibility. Replay the MetroPT-3 CSV as a real-time stream, publishing one RawDataDTO per row to raw_data_stream.
Contracts. Input: none (reads CSV from disk). Output: raw_data_stream → RawDataDTO.
State and lifecycle. Stateless. Reads the CSV once at startup, iterates rows in timestamp order, publishes each row with a sleep between publishes to control replay speed. When the CSV is exhausted, the generator loops back to the start (with an optional timestamp offset per loop, so the wall-clock replay never appears to go backward).
Key mechanic. Speed control via REPLAY_SPEED multiplier. A value of 600 means the pipeline consumes 6 hours of data every real-world minute — enough to reach the April 18 failure window within ~15 minutes of wall-clock time from a fresh start. A value of 1.0 runs in real time. A value of 1e9 runs as fast as Kafka accepts messages, useful for stress-testing consumers.
Configuration.
•	REPLAY_SPEED = 600 — replay acceleration factor.
•	LOOP = True — whether to loop when the CSV ends.
Trade-offs. The looped replay is deliberately naive (start over from row 0, optionally with time offset). A more sophisticated generator would maintain phase continuity or simulate degradation more realistically. The choice here reflects the demo goal: honest replay of real data rather than synthetic augmentation.
6.2 DataGate
Responsibility. Validate, clip, quality-flag, and label incoming raw rows, producing well-typed ProcessedRawDataDTO messages.
Contracts. Input: raw_data_stream → RawDataDTO. Output: processed_raw_data → ProcessedRawDataDTO.
State and lifecycle. Stateless per message. Configuration (sensor ranges, failure windows) is loaded at startup.
Key mechanics.
•	Validation. Missing sensors flagged; nulls replaced with the sensor's mid-range value; the row is flagged in quality_flags rather than dropped. Downstream services see a continuous stream, and quality information travels with the data.
•	Clipping. Values outside SENSOR_RANGES are clipped to bounds and flagged. Alternative would be to drop out-of-range rows entirely, but that breaks time series continuity for downstream rolling windows.
•	Labeling. For each row, DataGate checks whether the timestamp falls within PRE_FAILURE_HOURS before or during any window in FAILURE_WINDOWS. If yes, label = 1; otherwise 0.
•	Digital validation. Digital channels (COMP, LPS, etc.) must be 0.0 or 1.0. Non-binary values are coerced to the nearest binary value and flagged.
Configuration.
•	SENSOR_RANGES — hard bounds per sensor for clipping.
•	FAILURE_WINDOWS — list of (start, end) datetimes documenting known failures.
•	PRE_FAILURE_HOURS = 6 — pre-failure horizon for label derivation.
•	SCHEMA_VERSION — bumped when the DTO shape changes; propagated in quality_flags.
Trade-offs. Clipping rather than dropping preserves stream continuity but subtly alters distributional properties. For a system whose next stage is distribution monitoring, this is important: DataGate must never introduce artificial drift. The clipping bounds are set generously enough that in normal operation, no rows are clipped; clipping only triggers on genuinely anomalous readings.
6.3 FeatureExtractor
Responsibility. Convert single-row observations into fixed-dimension feature vectors that summarize behavior over a time window.
Contracts. Input: processed_raw_data → ProcessedRawDataDTO. Output: feature_vectors → FeatureVectorDTO.
State and lifecycle. Maintains a deque of the last N seconds of observations per analog sensor. On each incoming message, appends the current value, evicts stale entries (older than WINDOW_SECONDS), and computes summary statistics.
Key algorithm. For each of 7 analog sensors, compute 4 rolling statistics (mean, std, min, max) over the trailing 30-second window. Add 8 digital sensors as instantaneous values. Result: a 36-dimensional feature vector.
The 30-second choice matters. It is long enough to capture the compressor cycle (which typically runs on a duty cycle of tens of seconds), short enough that feature values track current behavior rather than distant history, and small enough that in-memory deques stay bounded (~30 floats per sensor × 7 sensors = 210 floats total state).
Configuration.
•	WINDOW_SECONDS = 30 — rolling window length.
•	FE_VERSION — bumped when the feature schema changes.
Trade-offs. The choice of statistics is deliberately conservative — mean/std/min/max are interpretable and cheap. More sophisticated features (spectral coefficients, autocorrelation, FFT bands, entropy) would likely improve model accuracy but at the cost of interpretability and computational overhead. The system architecture makes upgrading the feature set straightforward: bump FE_VERSION, retrain, downstream services see the change through the version stamp.
6.4 PrimaryModel
Responsibility. Serve failure-probability predictions on incoming feature vectors, and hot-reload its model when TransferLearning promotes a new one.
Contracts. Input: feature_vectors → FeatureVectorDTO and model_update_complete → ModelUpdateCompleteDTO. Output: predictions → PredictionDTO.
State and lifecycle. Stateless per prediction, but stateful across the model artifact. At startup, loads the latest primary model artifact from StorageManager. On each model_update_complete message where promotion_decision == "promoted", reloads the artifact — atomically, without dropping in-flight predictions. If no artifact exists at startup, the service logs a warning and silently drops feature vectors until a model is available.
Key mechanic. The model artifact is a dict containing three things: the fitted XGBoost classifier, the ordered list of feature names it was trained on, and the training metrics. Predictions align the incoming feature dictionary to the trained feature order before inference — so if the feature schema evolves (fe_version bumps), the model continues to work on features it recognizes and silently zero-imputes any it doesn't.
Model choice: XGBoost. Chosen for four reasons:
•	Handles imbalanced classes natively via scale_pos_weight.
•	Fast inference (microseconds per row).
•	Robust to unscaled features.
•	Interpretable via feature importance.
Alternatives considered: logistic regression (baseline reference, gave AUPRC ~0.4 — insufficient); random forest (comparable AUPRC to XGBoost, slower inference); shallow neural network (marginal improvement, opaque, harder to deploy). XGBoost is the sensible default and would need a real reason to abandon.
Configuration.
•	MODEL_PRED_THRESHOLD = 0.5 — probability threshold for label assignment.
•	Training script hyperparameters (n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight computed from class balance).
Trade-offs. The 0.5 threshold is naive and not calibrated to the operational cost trade-off. A production system would tune this based on the relative cost of false positives (unnecessary maintenance visits) versus false negatives (missed failures). The training script emits AUPRC, so re-thresholding by expected cost is straightforward once cost information is available.
6.5 DimReducer
Responsibility. Project 36-dimensional feature vectors to 2D coordinates for operator visualization.
Contracts. Input: feature_vectors → FeatureVectorDTO. Output: coordinates_nd → Coordinates2DDTO.
State and lifecycle. Bootstrap-then-serve. On first startup with no saved reducer, buffers the first 500 feature vectors. When the buffer fills, fits a Pipeline([StandardScaler, PCA(n_components=2)]), persists the artifact, and clears the buffer. Subsequent messages are transformed and published. On restart, the saved reducer is loaded and bootstrap is skipped.
Key mechanic. The StandardScaler is critical. Without it, PCA on 36-dim rolling statistics gives >99% variance on the first component because Oil_temperature (values ~60) dominates Motor_current_std (values ~0.001) numerically. With scaling, both features contribute proportionally to their information content, and typical explained variance for 2 components is 40–70% — meaningful summarization, not a numerical accident.
Configuration.
•	DR_BOOTSTRAP_SIZE = 500 — samples collected before fitting.
•	DR_N_COMPONENTS = 2 — output dimensionality.
Trade-offs. PCA is a linear method. Nonlinear alternatives (UMAP, t-SNE) would give more visually distinct clusters but at three real costs: non-deterministic output, no clean transform for out-of-sample points, and much slower per-message inference. For a live operator dashboard, PCA's determinism and speed are the right choice. The visual limitations of linear projection are acceptable given that the primary purpose is orientation ("is this point in the healthy cluster?"), not fine-grained classification.
6.6 DistributionMonitor
Responsibility. Detect when the live feature distribution has drifted from a baseline, and emit shift_events with severity classification.
Contracts. Input: feature_vectors → FeatureVectorDTO. Output: shift_events → ShiftEventDTO.
State and lifecycle. Two-phase. Phase 1: bootstrap. Collects the first 1000 feature vectors, computes per-feature summary statistics (empirical CDF via sorted values, retained in memory), persists the baseline. Phase 2: monitor. Maintains a rolling window of the last 500 feature vectors. On each new message, compares the current window to the baseline via Kolmogorov-Smirnov two-sample tests, per feature.
Key algorithm — KS test with top-k aggregation.
For each of 36 features, compute the KS statistic between the current 500-message window and the 1000-message baseline. This gives 36 per-feature drift scores in [0, 1]. The final drift score is the mean of the top-K (K = 5 by default) most-drifting features. This aggregation is important: averaging over all 36 features would dilute a strong signal in a subset of features; taking only the top-1 would be noisy. Top-K averaging balances sensitivity and stability.
Severity classification with hysteresis. Three thresholds:
Severity	Threshold	Meaning
warn	≥ 0.15	Distribution has shifted noticeably; watch.
alert	≥ 0.25	Significant drift; investigate.
critical	≥ 0.40	Substantial drift; retraining warranted.
Hysteresis is applied via cooldowns: once an event of severity S is emitted, no further event of the same severity is emitted for DM_COOLDOWN_MESSAGES (default 300) subsequent messages. This prevents alert storms during sustained drift. Higher-severity events reset the cooldown counter of lower severities — an alert following a warn always fires.
Interpretability. Each shift_events message includes the top-K feature names and their KS statistics in the window.top_features field. When drift fires, an operator can immediately see which sensors are drifting most, providing diagnostic context beyond the scalar severity.
Configuration.
•	DM_BOOTSTRAP_SIZE = 1000 — baseline sample count.
•	DM_WINDOW_SIZE = 500 — current window size.
•	DM_TOP_K = 5 — features to aggregate for final score.
•	DM_THRESHOLDS = {warn: 0.15, alert: 0.25, critical: 0.40} — severity thresholds.
•	DM_COOLDOWN_MESSAGES = 300 — messages between events of same severity.
Trade-offs. KS is a distributional test with well-understood statistical properties, computable in O(n log n) per feature, and requires no assumptions about the underlying distributions. Its main limitation is that it treats each feature independently — it can miss drift patterns where individual features look stationary but their joint distribution shifts. Alternatives like Population Stability Index or maximum mean discrepancy have similar per-feature semantics; multivariate tests (energy statistic, kernel two-sample) would catch joint drift but at significantly higher computational cost. The single-feature KS is the pragmatic default; the architecture makes replacement straightforward if better tests are needed.
6.7 TransferLearning
Responsibility. On drift events of sufficient severity, retrain a candidate model on the most recent labeled data, evaluate it against the incumbent, and hot-swap if the candidate improves.
Contracts. Input: shift_events → ShiftEventDTO and processed_raw_data → ProcessedRawDataDTO. Output: model_update_complete → ModelUpdateCompleteDTO.
State and lifecycle. Maintains a rolling deque of the last 20,000 processed rows (the recent buffer). On each shift_events message, evaluates three gates in order:
1.	Severity gate. event.severity ∈ TL_TRIGGER_SEVERITY (default: {critical} only).
2.	Cooldown gate. At least TL_RETRAIN_COOLDOWN_SECS (default 120s wall-clock) since the previous retrain attempt.
3.	Data gate. At least TL_MIN_POSITIVES (default 100) labeled positive rows in the recent buffer.
If all three pass, the retrain runs. If any fails, the event is logged and skipped.
Retrain procedure.
1.	Feature extraction. Recreate a FeatureExtractorService(connect=False) instance and replay the buffered rows through it. This uses the exact same feature-extraction code as the live pipeline — no training/serving skew.
2.	Split. Stratified 75/25 train/test split of the resulting feature vectors and labels.
3.	Train. Fit XGBoost with scale_pos_weight computed from the train split.
4.	Evaluate. Compute AUROC and AUPRC on the held-out test split.
5.	Compare. Load the incumbent model artifact, align its feature schema to the test split, and compute the incumbent's AUPRC on the same held-out data.
6.	Promotion decision. If candidate AUPRC ≥ incumbent AUPRC + TL_PROMOTION_MARGIN (default 0.02), promote. Otherwise, reject.
Artifact handling. Both promoted and rejected candidates are persisted:
•	Promoted → saved as a new primary model artifact. PrimaryModel's next model_update_complete triggers reload.
•	Rejected → saved as a primary_rejected artifact for post-hoc analysis. The model_update_complete message is still emitted (with promotion_decision = "rejected"), so the dashboard records the attempt.
Configuration.
•	TL_TRIGGER_SEVERITY = {"critical"} — which severities trigger retraining.
•	TL_RETRAIN_COOLDOWN_SECS = 120 — minimum wall-clock between retrains.
•	TL_RECENT_BUFFER_SIZE = 20000 — rows retained for retraining.
•	TL_MIN_POSITIVES = 100 — required labeled positives to proceed.
•	TL_PROMOTION_MARGIN = 0.02 — AUPRC improvement required for promotion.
Trade-offs.
•	Promotion margin. A 0.02 margin means we require statistically meaningful improvement, not any improvement. This prevents ping-pong promotion where minor stochastic differences trigger constant swaps. The cost is that occasional genuinely useful improvements are missed.
•	Positive threshold. Requiring 100 positives means that during long healthy stretches, drift events don't trigger retraining — which is correct behavior (a shift without failure examples cannot teach the model anything useful), but can feel counterintuitive to operators watching drift alerts fire with no model updates. Lowering the threshold to see updates faster is documented as a demo option.
•	Wall-clock cooldown. 120 seconds is generous. Retraining takes 5–15 seconds; the cooldown ensures the system doesn't spend most of its time retraining during sustained drift. In production, a shorter cooldown coupled with a longer promotion margin might be preferable.
________________________________________
7. The SSE bridge
7.1 Why a bridge exists
Kafka is a TCP-based protocol. Browsers speak HTTP (and WebSocket). The gap is architectural, not a technicality: Kafka clients expect long-lived connections, consumer group coordination, and message-level offset tracking. Browsers cannot participate in any of that. A bridge is required to translate.
The bridge implements Server-Sent Events (SSE), a one-directional HTTP streaming protocol standardized in the WHATWG HTML spec. SSE was chosen over WebSockets for four reasons:
•	One-directional is sufficient. The dashboard only reads; it doesn't send commands back. WebSockets are bidirectional; SSE is one-directional; the simpler protocol matches the actual requirement.
•	Native browser API. EventSource is built into every modern browser. No client library needed.
•	Auto-reconnect. The browser's EventSource automatically reconnects on connection loss with exponential backoff, without any application code.
•	HTTP-friendly. SSE flows over standard HTTP. It passes through firewalls, proxies, and load balancers without special configuration.
7.2 Fanout topology
per-client async queueper-client async queueper-client async queueSSE streamSSE streamSSE streampredictionsKafka consumer threadcoordinates_ndshift_eventsmodel_update_completeClient 1 queueClient 2 queueClient N queueBrowser 1Browser 2Browser N
The design factor is that Kafka consumers are relatively heavyweight — creating and destroying them per browser connection would exhaust broker resources at scale. Instead, the bridge maintains a single Kafka consumer running in a background thread. Each connected browser gets its own asyncio.Queue in the bridge process. The Kafka thread fans out each incoming message to every connected queue, and each browser reads from its own queue via a persistent SSE HTTP response.
7.3 Backpressure policy
Each per-client queue has a bounded capacity (QUEUE_MAXSIZE = 500). If a browser client is slow (paused on a debugger, throttled by the OS, on a slow network), its queue fills up. The bridge's response to a full queue is to drop messages for that client, silently, using put_nowait with an except QueueFull: pass.
The rationale is that a real-time dashboard should degrade gracefully rather than block. A stuck browser must never back-pressure the Kafka thread — that would break all other connected clients. Dropped messages for one client are invisible to other clients. The cost is that a debugging pause can result in visual gaps in the affected browser; the alternative (blocking) is unacceptable at scale.
7.4 Consumer offset semantics
The bridge uses auto_offset_reset="latest" with an ephemeral consumer group (UUID-suffixed group ID that changes per bridge process). This means:
•	On startup, the bridge sees only messages arriving after it connects. Historical messages already in Kafka are not replayed.
•	Restarting the bridge creates a fresh consumer group, so there is no "committed offset" state to recover from.
This is the correct semantic for a live dashboard: browsers want current data, not history. If a browser reloads, it sees the tail of the stream from that moment forward. The trade-off is that browsers connecting between failures see no data at all — but that is the honest answer to "what is happening right now?" when the answer is "nothing."
________________________________________
8. React dashboard
8.1 Component tree

<b> App.tsx </b>useSSE hookStreamState provider<b> Header </b>Title + connection pill<b> StatusBar </b>5 metric cards<b> Row 1 </b><b> Row 2 </b><b> Row 3 </b>Predictions/sCoords/sStatus pillModel versionUpdates count<b> ScatterChart </b>PCA 2D + prob color<b> ProbabilityChart </b>P(failure) timeline<b> DriftChart </b>Score + threshold refs<b> EventsTable </b>Recent drift events<b> UpdatesTable </b>Promote/reject decisions
8.2 State model
The root App component owns all dashboard state via a single custom hook, useSSE. This hook:
•	Opens an EventSource to http://localhost:8000/stream.
•	Parses each SSE event as JSON.
•	Routes messages to per-topic buffers (deque-style ring buffers, capped at 2000 messages per topic).
•	Maintains per-topic counters (total messages received per topic).
•	Exposes a connected boolean for connection status.
The hook returns a StreamState object containing all four buffers, all four counters, and the connected flag. Child components receive this object as a prop and read the slices they need. No component holds its own copy of stream data; the single source of truth is App's state.
8.3 Rate computation
Throughput (messages per second) is computed by a secondary hook, useRate, which samples a counter at 1-second intervals and computes the rate over a rolling 10-second window (newest_count - oldest_count) / window_seconds). The interval-and-rolling-window pattern avoids two failure modes: instantaneous rate (jittery) and cumulative average (never updates).
8.4 Chart library — Recharts
Recharts was chosen over the more common Plotly.js for two reasons:
•	React-idiomatic composition. Recharts uses actual React components (<LineChart>, <XAxis>, <Line>) as its declarative language, mirroring the rest of the UI. Plotly.js requires imperative configuration objects.
•	Bundle size. Recharts is lighter than Plotly.js by roughly an order of magnitude.
The three charts (probability line, drift score line + colored dots, PCA scatter) all use ResponsiveContainer for fluid width, isAnimationActive={false} to prevent visual thrashing during streaming updates, and consistent color/typography for visual coherence.
8.5 Coordinate/prediction alignment
The scatter chart color-codes each 2D point by the corresponding prediction probability. Since coordinates and predictions arrive on separate topics with different rhythms, the join is done client-side. The current implementation builds a Map<ts, prob> from the predictions buffer once per render (memoized via useMemo), then looks up each coordinate's probability in O(1). Missing predictions (e.g., due to bridge queue drops) fall back to prob = 0.
An alternative would be pd.merge_asof-style nearest-timestamp joining. This was used in the Streamlit dashboard but is unnecessary here because coordinates and predictions share the input feature_vectors and therefore their timestamps match exactly.
________________________________________
9. Orchestration
The orchestrator.py script is a lightweight Python supervisor that:
•	Starts the 7 services in dependency order, with a 1.5-second delay between each to allow Kafka consumer group assignment to stabilize.
•	Prefixes each service's stdout with a colored tag ([gate], [fe], [model], etc.) so interleaved logs remain readable.
•	Handles SIGINT (Ctrl+C) by terminating all services in reverse order with a grace period.
•	Exits with a nonzero code if any service dies unexpectedly.
Why not Docker Compose for services? The seven services are Python processes with shared code and a shared virtual environment. Containerizing each would require seven identical Docker images differing only in the entry point, seven volume mounts for the artifact directory, and additional network configuration. For a development-oriented system, the orchestrator is materially simpler with no cost to reproducibility (all services still speak to Kafka over TCP, which is containerized). For a production deployment, the situation reverses: containerization becomes the correct answer.
Why not a real supervisor (systemd, supervisord, PM2)? Same reason. Real supervisors handle process lifecycle across reboots, log rotation, resource limits, and dependency management — all irrelevant for local development. For local orchestration, 60 lines of Python is the right tool.
________________________________________
10. Storage and versioning
The StorageManager class in metropt/storage.py provides a single API for persisting artifacts. Three artifact types are supported:
•	Models. XGBoost pipelines, saved as joblib dumps of {model, feature_names, metrics} dicts.
•	Baselines. DistributionMonitor's frozen feature statistics, saved as pickled numpy arrays.
•	Reducers. DimReducer's PCA + scaler pipeline.
Each artifact is namespaced by (project, model_type). Filenames incorporate a UTC timestamp: metropt/primary/model_20260701_101149.joblib. Loading defaults to the latest version by lexicographic sort; explicit versions can be requested for rollback or debugging.
Atomicity via file naming. When TransferLearning writes a new model, it writes to a new filename, not by overwriting the existing one. PrimaryModel reads the latest version at reload time. There is no half-written state: either the new file exists complete or it doesn't. This is the same pattern used by write-then-rename atomic updates in databases and package managers, and it works here because filesystem writes are large-block atomic.
Storage layout.
data/artifacts/
├── models/
│   └── metropt/
│       ├── primary/
│       │   └── model_20260701_101149.joblib
│       ├── primary_rejected/
│       │   └── model_20260702_150233.joblib
│       ├── dm/
│       │   └── model_20260701_114753.joblib
│       └── dr/
│           └── model_20260701_112733.joblib
└── training/
    └── ... (persisted feature vectors used by training script)
________________________________________
11. Testing and CI
11.1 Test layers
The test suite is organized into three layers, each answering a different question.
Layer 1: DTO contract tests. These are trivial-looking round-trip tests: construct a DTO with representative data, serialize to JSON, deserialize back, assert equality. They catch every schema drift immediately. For each of the 7 DTOs, plus edge cases (probability out of range → ValidationError, invalid severity string → ValidationError). ~5 lines per test.
Layer 2: Service handler tests. For each service, mock the Kafka producer and consumer, call handle() directly with test messages, assert on the produced output. Key patterns:
•	The connect: bool = True parameter in BaseService.__init__ allows constructing services with connect=False, which skips all Kafka wiring. This makes offline testing trivial and — importantly — makes the training script possible (it composes DataGate and FeatureExtractor as regular Python objects, no Kafka needed).
•	The _last_published instance attribute (initialized in BaseService.__init__) records what would have been published when the service is running in connect=False mode. Tests read this to verify the output DTO.
Layer 3: Integration tests. End-to-end pipeline tests that start the full stack in a subprocess, run a small CSV slice, and assert on the emitted messages. These are slower (30+ seconds) but catch inter-service issues that unit tests miss. Currently limited in scope; the aspiration is broader coverage as the system stabilizes.
11.2 CI pipeline
GitHub Actions on every push:
yaml
jobs:
  test:
    steps:
      - checkout
      - install uv
      - uv sync
      - uv run pytest tests/
      - uv run ruff check .
The tests currently cover ~85% of service handler logic. Gaps are primarily in error paths (malformed messages, Kafka connection failures) that are hard to trigger in unit tests but are covered by defensive coding.
________________________________________
12. Results and observations
12.1 Training metrics
Trained on 202,259 rows sampled uniformly from the full CSV, with PRE_FAILURE_HOURS = 6 labeling:
Metric	Value
Positive class rate	2.52%
scale_pos_weight	38.6
Train / Test split	161,807 / 40,452
AUROC	0.985
AUPRC	0.899
The AUPRC is 36× the random baseline (0.899 vs. 0.025). This is genuinely strong performance, but comes with the leakage caveat noted in §3.3: the random split allows rows from within a failure event's pre-window to appear on both sides of the split, and their shared rolling-window state gives an optimistic estimate. A split-by-failure evaluation would be lower — likely in the 0.6–0.8 range — and would be closer to production reality.
12.2 Observed drift behavior
During live operation, DistributionMonitor typically fires:
•	In pre-failure windows. As expected: features drift as the failure approaches. Top drifting features consistently include Motor_current (compressor working harder), TP3/Reservoirs (pressure setpoints changing), and Oil_temperature (thermal signature of extended operation).
•	In extended healthy stretches. Occasionally, driven by sensor noise or environmental changes. These fire less frequently and typically at warn or alert severity rather than critical.
•	Immediately after retraining. If TransferLearning promotes a new model, the drift monitor's next window is drawn on new data, but the baseline is unchanged. This can cause a brief post-promotion warn event that self-resolves as the window catches up.
12.3 TransferLearning behavior
Under default settings (TL_MIN_POSITIVES = 100), TransferLearning triggers infrequently — typically zero to a few times per hour of wall-clock. This is expected: during healthy stretches, drift events fire but the recent buffer contains no positives, and TransferLearning correctly refuses to retrain. Real retraining happens on failure days, and the promotion rate is roughly 50/50 — some candidates beat the incumbent by the required margin, others don't.
For demo purposes, TL_MIN_POSITIVES = 10 is useful: it lets a viewer see retraining trigger within minutes of startup and observe both promoted and rejected outcomes.
________________________________________
13. Trade-offs and known limitations
13.1 Random-split leakage in offline training
The training script uses stratified random splits. As discussed in §3.3 and §12.1, this inflates the reported AUPRC compared to a rigorous split-by-failure evaluation. The system architecture makes it easy to fix: the training script could accept a --split-by-failure flag, group rows by their nearest failure window, and train/test on disjoint groups. This would be the right change for a rigorous evaluation but was not the focus of the current build.
13.2 KS thresholds are fixed
The DM_THRESHOLDS for warn/alert/critical are hardcoded and not derived from data. In principle, they should be set adaptively — for instance, as percentiles of the KS score distribution during a known-healthy period. This would make the system self-calibrating to different sensor suites and data volumes. The current approach is simpler but requires manual tuning per deployment.
13.3 Single Kafka broker
The system uses a single-broker Kafka setup via Docker Compose. This is fine for development and single-user demonstration but is not a highly available production configuration. For real deployment, a 3-node Kafka cluster with replication factor 3 would be the minimum. The application-level code needs no changes; only Docker Compose (or Kubernetes) configuration.
13.4 Orchestration is process-supervisory only
The orchestrator.py script starts and stops processes but does no restart-on-failure. If a service crashes, the orchestrator logs the exit code and terminates the entire pipeline. In production, per-service restart with backoff would be handled by a container runtime or systemd. For a demo, the current behavior (fail loudly) is preferable to fail silently.
13.5 Adaptation cadence assumes labels arrive
TransferLearning assumes that labeled data is available in the recent buffer. In this system, labels are derived at ingestion time from known failure windows — a luxury of retrospective analysis on a well-documented dataset. In production, labels typically arrive with latency (a maintenance technician confirms a fault hours or days after it happens). Adapting the system to this reality would require a separate label ingestion path — perhaps a dedicated Kafka topic for maintenance events that DataGate consumes to update labels retroactively.
13.6 The dashboard's coordinate/prediction join
The React dashboard's join between coordinates and predictions assumes exact timestamp matches. This is correct for the current pipeline because both are derived from the same feature vector. If a future architecture change decoupled them (e.g., a separate coordinate producer), the join would need to become tolerance-based via a nearest-neighbor lookup.
________________________________________
14. What I'd build next
Split-by-failure evaluation. The single most impactful change for honest metrics.
Adaptive drift thresholds. Compute warn/alert/critical cutoffs as percentiles of the KS score distribution during a calibrated healthy period. Would make the system self-tuning across deployments.
Multi-tool support. The DTO schema already includes product_id and tool_id fields. Extending the pipeline to handle multiple simultaneous units would require per-tool state in FeatureExtractor, DistributionMonitor, and TransferLearning — but no schema changes and no cross-service coordination changes.
Better feature engineering. The current 4-statistic rolling window is deliberately simple. Adding spectral features (FFT band energies), autocorrelation, and cross-sensor correlation coefficients would likely improve classification accuracy. Whether this is worth the interpretability trade-off depends on operator needs.
Kubernetes packaging. For production deployment, the seven services become Deployments with autoscaling, Kafka becomes a Strimzi cluster, and the SSE bridge scales horizontally behind an Ingress. This is straightforward infrastructure work with no application code changes.
Model rollback UI. Currently, model versions are timestamp-named and rollback is a manual filesystem operation. A dashboard control that lets an operator pin an earlier model version (with an emitted model_update_complete event to notify the pipeline) would close this loop.
Alerting integration. Critical-severity shift_events should page an operator. A small consumer service that translates shift_events to PagerDuty/Slack/email would be trivial to add — one more service consuming one topic.
________________________________________
15. References and appendices
15.1 References
Veloso, B., Ribeiro, R. P., Gama, J. & Pereira, P. M. The MetroPT dataset for predictive maintenance. Scientific Data 9, 764 (2022).
Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M. & Gama, J. Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry. IEEE DSAA (2021).
Chen, T. & Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD 2016.
Massey, F. J. The Kolmogorov-Smirnov Test for Goodness of Fit. Journal of the American Statistical Association 46 (1951).
WHATWG. HTML Living Standard, Section 9.2: Server-sent events.
15.2 Repository layout
metropt/
├── config/
│   ├── __init__.py
│   └── settings.py         # All configuration constants
├── dtos.py                 # Pydantic contracts (7 DTOs)
├── kafka_utils.py          # Producer/consumer factories
├── storage.py              # StorageManager (artifact persistence)
├── services/
│   ├── base.py             # BaseService (consume loop, publish helper)
│   ├── data_generator.py
│   ├── data_gate.py
│   ├── feature_extractor.py
│   ├── primary_model.py
│   ├── dim_reducer.py
│   ├── distribution_monitor.py
│   └── transfer_learning.py
├── scripts/
│   ├── create_topics.py    # Kafka topic setup
│   ├── train_primary.py    # Offline model training
│   └── orchestrator.py     # Process supervisor
└── ui/
    └── dashboard.py        # Streamlit dashboard

bridge/
├── main.py                 # FastAPI + SSE
└── pyproject.toml

frontend/
├── src/
│   ├── App.tsx
│   ├── types/dtos.ts       # TypeScript mirrors of Pydantic DTOs
│   ├── hooks/
│   │   ├── useSSE.ts
│   │   └── useRate.ts
│   └── components/
│       ├── StatusBar.tsx
│       ├── ProbabilityChart.tsx
│       ├── ScatterChart.tsx
│       ├── DriftChart.tsx
│       ├── EventsTable.tsx
│       ├── UpdatesTable.tsx
│       └── ui/             # shadcn primitives (button, card, etc.)
├── package.json
└── vite.config.ts

tests/
├── test_dtos.py
├── test_data_gate.py
├── test_feature_extractor.py
├── test_primary_model.py
├── test_dim_reducer.py
├── test_distribution_monitor.py
└── test_transfer_learning.py

.github/workflows/
└── ci.yml                  # Test + lint on every push
15.3 Configuration reference
Full reference of environment-configurable settings (all in metropt/config/settings.py, overridable via PRISM_* environment variables):
Setting	Default	Purpose
REPLAY_SPEED	600	CSV replay acceleration factor
PRE_FAILURE_HOURS	6	Labeling window before failures
SCHEMA_VERSION	"v1.0"	DataGate output schema version
FE_WINDOW_SECONDS	30	FeatureExtractor rolling window
FE_VERSION	"v1.0"	Feature extractor version stamp
MODEL_PRED_THRESHOLD	0.5	Probability threshold for label 1
DR_BOOTSTRAP_SIZE	500	Samples before PCA is fit
DR_N_COMPONENTS	2	PCA output dimensionality
DM_BOOTSTRAP_SIZE	1000	Samples in drift baseline
DM_WINDOW_SIZE	500	Rolling current window
DM_TOP_K	5	Features aggregated for score
DM_THRESHOLDS	{warn: 0.15, alert: 0.25, critical: 0.40}	Severity tiers
DM_COOLDOWN_MESSAGES	300	Messages between same-severity events
TL_TRIGGER_SEVERITY	{"critical"}	Severities that trigger retraining
TL_RETRAIN_COOLDOWN_SECS	120	Wall-clock between retrains
TL_RECENT_BUFFER_SIZE	20000	Rows retained for retraining
TL_MIN_POSITIVES	100	Positives required to retrain
TL_PROMOTION_MARGIN	0.02	AUPRC delta for promotion
________________________________________
End of report.
