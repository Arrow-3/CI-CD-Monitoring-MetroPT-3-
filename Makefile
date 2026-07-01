.PHONY: infra topics explore run-gen consume test
infra:    ; docker compose up -d
topics:   ; uv run python -m metropt.scripts.create_topics
explore:  ; uv run python -m metropt.scripts.explore_data
run-gen:  ; uv run python -m metropt.services.data_generator
consume:  ; uv run python -m metropt.scripts.consume_raw
run-gate:  ; uv run python -m metropt.services.data_gate
consume-processed: ; uv run python -m metropt.scripts.consume_processed
run-fe:           ; uv run python -m metropt.services.feature_extractor
consume-features: ; uv run python -m metropt.scripts.consume_features
train:            ; uv run python -m metropt.scripts.train_primary
run-model:        ; uv run python -m metropt.services.primary_model
consume-predictions: ; uv run python -m metropt.scripts.consume_predictions
saved-models: uv run python -c "from metropt.storage import StorageManager; sm = StorageManager(); print(sm.latest_model_version('metropt', 'primary'))"
run-dr:         ; uv run python -m metropt.services.dim_reducer
consume-coords: ; uv run python -m metropt.scripts.consume_coords
test:     ; uv run pytest -v tests



# Saved trained models from powershell: 	ls data/artifacts/models/metropt/primary 