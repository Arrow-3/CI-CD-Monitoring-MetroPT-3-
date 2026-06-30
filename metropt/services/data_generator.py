"""Replays MetroPT-3 CSV sensor history to Kafka as validated RawDataDTO messages at configurable speed.
Simulates live MetroPT sensor feeds by replaying archived CSV data into Kafka."""

from __future__ import annotations
import logging, time
import pandas as pd
from metropt import kafka_utils
from metropt.config import settings
from metropt.dtos import RawDataDTO

class DataGeneratorService:
    name = "datageneratorservice"

    def __init__(self, path=settings.DATA_PATH, speed=settings.REPLAY_SPEED):
        self.path, self.speed = path, speed
        self.producer = kafka_utils.get_producer()
        self.log = logging.getLogger(self.name)

    def load(self) -> pd.DataFrame:
        #df = pd.read_csv(self.path)                     # It consumes lots of memory by pyarrow engine
        return pd.read_csv(self.path, chunksize=5000)    # It consumes less memory by standard engine

    def run(self) -> None:
        chunks = self.load()
        for df in chunks:
            # Clean and transform the current 5,000-row chunk
            if df.columns[0].lower().startswith("unnamed"):     # Filters out all the unnamed instances
                df = df.drop(columns=df.columns[0])

            df["timestamp"] = pd.to_datetime(df["timestamp"])   # Transforms timestamps in accordance to index values
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Extract columns to stream
            cols = [c for c in settings.ANALOG_COLS + settings.DIGITAL_COLS if c in df.columns]
            self.log.info("streaming %d rows, %d sensors, %.0fx", len(df), len(cols), self.speed)
            delay = 1.0 / self.speed  

            # Stream individual rows to Kafka
            for _, row in df.iterrows():
                dto = RawDataDTO(
                    ts=row["timestamp"].isoformat(),
                    product_id="APU", 
                    tool_id="metro-1",
                    raw={c: float(row[c]) for c in cols},
                )
                self.producer.send("raw_data_stream", dto.to_json())
                time.sleep(delay)   

            # Flush at the end of each chunk to keep the Kafka buffer clean
            self.producer.flush()


if __name__ == "__main__":
    DataGeneratorService().run()
