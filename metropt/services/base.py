from __future__ import annotations
import abc, logging
from metropt import kafka_utils
from metropt.config.settings import get_consumer_group_id

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

class BaseService(abc.ABC):
    name: str = "base"
    input_topic: str | None = None

    def __init__(self, *, connect: bool = True):
        self.log = logging.getLogger(self.name)
        self.producer = None
        self.consumer = None
        self._last_published = None  # To ensure the smooth run for the training as it starts from : None
        if connect:
            self._connect_kafka()

    def _connect_kafka(self) -> None:
        self.producer = kafka_utils.get_producer()
        if self.input_topic:
            self.consumer = kafka_utils.get_consumer(
                self.input_topic, get_consumer_group_id(self.name))

    def publish(self, topic: str, dto) -> None:
        if self.producer is None:
            self._last_published = dto       # offline mode / only reached when connect=False
            return
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