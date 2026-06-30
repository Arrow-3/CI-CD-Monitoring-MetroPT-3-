import uuid
from metropt import kafka_utils
from metropt.config.settings import get_consumer_group_id

# Fresh group every run → no committed offset → honors "earliest"
group = get_consumer_group_id(f"debug_processed_{uuid.uuid4().hex[:8]}")
c = kafka_utils.get_consumer("processed_raw_data", group)

print(f"consumer group: {group}")
print("waiting for messages (Ctrl-C to stop)...")

count = 0
for msg in c:                      # blocks waiting for messages
    print(msg.value)
    count += 1
    if count >= 5:
        break

print(f"done — read {count} messages")

'''
from metropt import kafka_utils
from metropt.config.settings import get_consumer_group_id

c = kafka_utils.get_consumer("processed_raw_data",
                             get_consumer_group_id("debug_processed"))
for i, msg in enumerate(c):
    print(msg.value)
    if i >= 4:
        break   '''