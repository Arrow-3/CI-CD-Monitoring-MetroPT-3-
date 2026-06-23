# CI-CD-Monitoring-MetroPT-3-

metropt/
├── pyproject.toml              # deps + tooling (Poetry)
├── docker-compose.yml          # Kafka + ZooKeeper
├── Makefile                    # shortcuts
├── .github/workflows/ci.yml    # CI/CD
├── data/                       # ← Include MetroPT-3 CSV lives here (gitignored)
├── metropt/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # topics, consumer-group rule, replay + labeling config
│   ├── dtos.py                 # the message contracts
│   ├── kafka_utils.py          # producer/consumer factories
│   ├── services/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseService every consumer inherits
│   │   └── data_generator.py   # streams the CSV → raw_data_stream
│   └── scripts/
│       ├── create_topics.py
│       ├── explore_data.py
│       └── consume_raw.py      # debug reader to verify the stream
└── tests/
    └── test_dtos.py



# Running the DataGenerator under uv
"""
uv sync                                         # creates .venv, installs everything
docker compose up -d                            # Kafka + ZooKeeper
uv run python -m metropt.scripts.explore_data     # confirm column names first
uv run python -m metropt.scripts.create_topics    # the 7 topics
uv run python -m metropt.services.data_generator  # terminal 1: stream MetroPT-3
uv run python -m metropt.scripts.consume_raw      # terminal 2: verify RawDataDTO flowing    """
