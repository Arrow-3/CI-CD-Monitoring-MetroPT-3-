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
test:     ; uv run pytest -v tests
