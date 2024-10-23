import json
from kafka_utils import get_producer, get_consumer, send_message, consume_messages
import logging
from datetime import datetime
from typing import Dict, Any
from core.avro_serialization import AvroSerializer

logger = logging.getLogger(__name__)

    def __init__(self):
        self.producer = get_producer()
        self.consumers = {}
        self.avro_serializer = AvroSerializer()
        self.retry_processor = RetryProcessor(self)
        self.topics = {
            'knowledge_transfer': 'besai-knowledge-transfer',
            'cognitive_update': 'besai-cognitive-update',
            'conflict_resolution': 'besai-conflict-resolution',
            'knowledge_update': 'besai-knowledge-update',
            'system_metrics': 'besai-system-metrics',
            'quinca_updates': 'quinca-updates'
        }
        self.retry_processor.start()

    def send_knowledge_transfer(self, knowledge_graph: Dict[str, Any]):
        """Send knowledge transfer using Avro serialization."""
        try:
            avro_bytes = self.avro_serializer.serialize_knowledge_transfer(knowledge_graph)
            message = {
                'type': 'knowledge_transfer',
                'encoding': 'avro',
                'data': avro_bytes.hex()  # Convert bytes to hex string for JSON serialization
            }
            self._send_to_topic('knowledge_transfer', message)
            logger.info(f"Successfully sent Avro-encoded knowledge transfer")
        except Exception as e:
            logger.error(f"Error sending knowledge transfer: {str(e)}")
            # Fallback to JSON if Avro fails
            self._send_json_fallback('knowledge_transfer', knowledge_graph)

    def send_cognitive_update(self, cognitive_state: Dict[str, Any], nlp_analysis: Dict[str, Any]):
        """Send cognitive update using Avro serialization."""
        try:
            avro_bytes = self.avro_serializer.serialize_cognitive_update(cognitive_state)
            message = {
                'type': 'cognitive_update',
                'encoding': 'avro',
                'data': avro_bytes.hex(),
                'nlp_analysis': nlp_analysis  # Keep NLP analysis as JSON
            }
            self._send_to_topic('cognitive_update', message)
            logger.info(f"Successfully sent Avro-encoded cognitive update")
        except Exception as e:
            logger.error(f"Error sending cognitive update: {str(e)}")
            # Fallback to JSON if Avro fails
            self._send_json_fallback('cognitive_update', {
                'cognitive_state': cognitive_state,
                'nlp_analysis': nlp_analysis
            })

    def consume_quinca_updates(self, callback):
        """Consume updates with Avro deserialization support."""
        topic = self.topics['quinca_updates']
        if topic not in self.consumers:
            self.consumers[topic] = get_consumer(topic)
        
        try:
            for message in consume_messages(self.consumers[topic]):
                logger.info(f"Received message from topic: {topic}")
                try:
                    if message.get('encoding') == 'avro':
                        # Decode Avro message
                        avro_bytes = bytes.fromhex(message['data'])
                        if message['type'] == 'knowledge_transfer':
                            data = self.avro_serializer.deserialize_knowledge_transfer(avro_bytes)
                        elif message['type'] == 'cognitive_update':
                            data = self.avro_serializer.deserialize_cognitive_update(avro_bytes)
                        else:
                            data = message['data']
                    else:
                        # Handle JSON message
                        data = message['data']
                    
                    processed_message = {
                        'type': message['type'],
                        'data': data,
                        'metadata': message.get('metadata', {})
                    }
                    callback(processed_message)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    self._handle_message_processing_failure(message, e)
        except Exception as e:
            logger.error(f"Error consuming messages from {topic}: {str(e)}")
            self._handle_consumer_failure(topic, e)

    def send_conflict_resolution_notification(self, 
                                           entity: str, 
                                           conflict_type: str, 
                                           resolution_strategy: str, 
                                           resolution_details: Dict[str, Any]):
        """Send notifications about conflict resolution events."""
        message = {
            'type': 'conflict_resolution',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'entity': entity,
                'conflict_type': conflict_type,  # 'attribute', 'relationship', or 'entity'
                'resolution_strategy': resolution_strategy,  # 'override', 'merge', or 'keep_both'
                'resolution_details': resolution_details,
                'metadata': {
                    'resolution_time': datetime.now().isoformat(),
                    'confidence': resolution_details.get('confidence', 1.0)
                }
            }
        }
        self._send_to_topic('conflict_resolution', message)

    def send_knowledge_update_notification(self, 
                                        update_type: str, 
                                        affected_entities: List[str], 
                                        changes: Dict[str, Any],
                                        version_info: Dict[str, Any]):
        """Send notifications about knowledge base updates."""
        message = {
            'type': 'knowledge_update',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'update_type': update_type,  # 'addition', 'modification', 'deletion'
                'affected_entities': affected_entities,
                'changes': changes,
                'version_info': version_info,
                'metadata': {
                    'update_time': datetime.now().isoformat(),
                    'update_source': changes.get('source', 'system'),
                    'update_certainty': changes.get('certainty', 1.0)
                }
            }
        }
        self._send_to_topic('knowledge_update', message)

    def send_system_metrics(self, 
                          metrics: Dict[str, Any], 
                          performance_data: Optional[Dict[str, Any]] = None):
        """Send system performance and metrics notifications."""
        message = {
            'type': 'system_metrics',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'metrics': metrics,
                'performance': performance_data or {},
                'metadata': {
                    'collection_time': datetime.now().isoformat(),
                    'metrics_version': '1.0'
                }
            }
        }
        self._send_to_topic('system_metrics', message)

    def _send_to_topic(self, topic_key: str, message: Dict[str, Any]):
        """Send message to a specific topic with retries."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                topic = self.topics[topic_key]
                send_message(self.producer, topic, message)
                logger.info(f"Successfully sent message to topic: {topic}")
                return
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Failed to send message after {max_retries} attempts: {str(e)}")
                    self._handle_send_failure(topic_key, message, e)
                else:
                    logger.warning(f"Retry {retry_count} for topic {topic_key}: {str(e)}")
                    time.sleep(1 * retry_count)  # Exponential backoff

    def _send_json_fallback(self, topic_key: str, data: Dict[str, Any]):
        """Send message using JSON encoding as fallback."""
        message = {
            'type': topic_key,
            'encoding': 'json',
            'data': data,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'fallback': True
            }
        }
        self._send_to_topic(topic_key, message)
        logger.warning(f"Used JSON fallback for {topic_key} message")

    def _handle_send_failure(self, topic_key: str, message: Dict[str, Any], error: Exception):
        """Handle failed message sending attempts with retry processor."""
        logger.error(f"Failed to send message to {topic_key}: {error}")
        self.retry_processor.add_failed_message(topic_key, message, str(error))

    def _handle_message_processing_failure(self, message: Dict[str, Any], error: Exception):
        """Handle failures in message processing."""
        logger.error(f"Failed to process message: {error}")
        # Implement dead letter queue logic
        self._send_to_dead_letter_queue(message, str(error))

    def _handle_consumer_failure(self, topic: str, error: Exception):
        """Handle consumer failures with reconnection logic."""
        logger.error(f"Consumer failure for topic {topic}: {error}")
        try:
            if topic in self.consumers:
                self.consumers[topic].close()
                del self.consumers[topic]
            # Implement reconnection logic here
            time.sleep(5)  # Wait before reconnecting
            self.consumers[topic] = get_consumer(topic)
        except Exception as e:
            logger.error(f"Error during consumer recovery for topic {topic}: {str(e)}")

    def _persist_failed_message(self, topic_key: str, message: Dict[str, Any]):
        """Persist failed messages for later retry."""
        try:
            # Implement persistence logic (e.g., to a file or database)
            # This is a simple file-based implementation
            failed_message = {
                'topic': topic_key,
                'message': message,
                'timestamp': datetime.now().isoformat(),
            }
            with open('failed_messages.json', 'a') as f:
                json.dump(failed_message, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error persisting failed message: {str(e)}")

    def _send_to_dead_letter_queue(self, message: Dict[str, Any], error: str):
        """Send failed messages to a dead letter queue."""
        dlq_message = {
            'original_message': message,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        try:
            # You could send to a special Kafka topic for dead letters
            dlq_topic = f"{self.topics.get(message['type'], 'unknown')}-dlq"
            send_message(self.producer, dlq_topic, dlq_message)
        except Exception as e:
            logger.error(f"Error sending to dead letter queue: {str(e)}")

    def close(self):
        """Clean up resources including retry processor."""
        try:
            self.retry_processor.stop()
            self.producer.close()
            for topic, consumer in self.consumers.items():
                try:
                    consumer.close()
                except Exception as e:
                    logger.error(f"Error closing consumer for topic {topic}: {str(e)}")
        except Exception as e:
            logger.error(f"Error closing Kafka client: {str(e)}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = BeSAIKafkaClient()
    
    # Example: Send a conflict resolution notification
    client.send_conflict_resolution_notification(
        entity="AI",
        conflict_type="attribute",
        resolution_strategy="merge",
        resolution_details={
            "attribute": "capability",
            "original_value": "machine learning",
            "conflicting_value": "deep learning",
            "resolved_value": ["machine learning", "deep learning"],
            "confidence": 0.9
        }
    )
    
    # Example: Send a knowledge update notification
    client.send_knowledge_update_notification(
        update_type="modification",
        affected_entities=["AI", "Machine Learning"],
        changes={
            "source": "user_input",
            "certainty": 0.95,
            "modifications": {
                "AI": {"capability": "deep learning"},
                "Machine Learning": {"relationship": "subset_of_AI"}
            }
        },
        version_info={
            "previous_version": 1,
            "new_version": 2,
            "update_type": "incremental"
        }
    )
    
    # Example: Send system metrics
    client.send_system_metrics(
        metrics={
            "knowledge_base_size": 1000,
            "relationship_count": 5000,
            "conflict_resolution_rate": 0.95,
            "average_certainty": 0.87
        },
        performance_data={
            "processing_time": 0.05,
            "memory_usage": "256MB",
            "active_connections": 10
        }
    )
    
    # Don't forget to close the client when done
    client.close()
