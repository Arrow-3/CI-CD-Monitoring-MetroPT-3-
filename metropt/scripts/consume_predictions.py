import uuid
from metropt import kafka_utils
from metropt.config.settings import get_consumer_group_id

group = get_consumer_group_id(f"debug_predictions_{uuid.uuid4().hex[:8]}")
c = kafka_utils.get_consumer("predictions", group)
print(f"consumer group: {group}\nwaiting for messages...")
count = 0
for msg in c:
    print(msg.value)
    count += 1
    if count >= 3:
        break
print(f"done — read {count} messages")  