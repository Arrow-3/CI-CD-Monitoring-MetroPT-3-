from metropt import kafka_utils
from metropt.config.settings import get_consumer_group_id

c = kafka_utils.get_consumer("raw_data_stream", get_consumer_group_id("debug_reader"))
for i, msg in enumerate(c):
    print(msg.value)
    if i >= 4:
        break
