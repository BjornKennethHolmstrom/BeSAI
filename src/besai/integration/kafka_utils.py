from kafka import KafkaProducer, KafkaConsumer
import json
import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

def get_producer():
    return KafkaProducer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def get_consumer(topic):
    return KafkaConsumer(
        topic,
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

def send_message(producer, topic, message):
    future = producer.send(topic, message)
    try:
        future.get(timeout=10)
    except Exception as e:
        print(f"Error sending message to Kafka: {e}")

def consume_messages(consumer):
    for message in consumer:
        yield message.value
