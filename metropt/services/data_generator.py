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
        df = pd.read_csv(self.path)
        if df.columns[0].lower().startswith("unnamed"):
            df = df.drop(columns=df.columns[0])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    def run(self) -> None:
        df = self.load()
        cols = [c for c in settings.ANALOG_COLS + settings.DIGITAL_COLS if c in df.columns]
        self.log.info("streaming %d rows, %d sensors, %.0fx", len(df), len(cols), self.speed)
        delay = 1.0 / self.speed
        for _, row in df.iterrows():
            dto = RawDataDTO(
                ts=row["timestamp"].isoformat(),
                product_id="APU", tool_id="metro-1",
                raw={c: float(row[c]) for c in cols},
            )
            self.producer.send("raw_data_stream", dto.to_json())
            time.sleep(delay)
        self.producer.flush()

if __name__ == "__main__":
    DataGeneratorService().run()
