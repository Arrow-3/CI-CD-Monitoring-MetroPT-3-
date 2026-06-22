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
