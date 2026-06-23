from kafka.admin import KafkaAdminClient, NewTopic
from metropt.config import settings

def main():
    admin = KafkaAdminClient(bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS)
    existing = set(admin.list_topics())
    new = [NewTopic(t, settings.NUM_PARTITIONS, settings.REPLICATION_FACTOR)
           for t in settings.TOPICS if t not in existing]
    if new:
        admin.create_topics(new)
        print("created:", [t.name for t in new])
    else:
        print("all topics already exist")
    admin.close()

if __name__ == "__main__":
    main()
