from __future__ import annotations
import abc, logging
from metropt import kafka_utils
from metropt.config.settings import get_consumer_group_id

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

class BaseService(abc.ABC):
    name: str = "base"
    input_topic: str | None = None

    def __init__(self):
        self.log = logging.getLogger(self.name)
        self.producer = kafka_utils.get_producer()
        if self.input_topic:
            self.consumer = kafka_utils.get_consumer(
                self.input_topic, get_consumer_group_id(self.name))

    def publish(self, topic: str, dto) -> None:
        self.producer.send(topic, dto.to_json())

    @abc.abstractmethod
    def handle(self, message: str) -> None: ...

    def run(self) -> None:
        self.log.info("starting; consuming %s", self.input_topic)
        for msg in self.consumer:
            try:
                self.handle(msg.value)
            except Exception:
                self.log.exception("error handling message")
