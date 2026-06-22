.PHONY: infra topics explore run-gen consume test
infra:    ; docker compose up -d
topics:   ; uv run python -m metropt.scripts.create_topics
explore:  ; uv run python -m metropt.scripts.explore_data
run-gen:  ; uv run python -m metropt.services.data_generator
consume:  ; uv run python -m metropt.scripts.consume_raw
test:     ; uv run pytest -v tests
