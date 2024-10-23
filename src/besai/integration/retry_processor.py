import json
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from queue import Queue, Empty
import signal

logger = logging.getLogger(__name__)

class RetryProcessor:
    def __init__(self, kafka_client, retry_interval: int = 300, max_retries: int = 3):
        """
        Initialize the retry processor.
        
        Args:
            kafka_client: The BeSAIKafkaClient instance
            retry_interval: Seconds between retry attempts (default: 5 minutes)
            max_retries: Maximum number of retry attempts per message
        """
        self.kafka_client = kafka_client
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.running = False
        self.retry_thread = None
        
        # Get base directory from logging config
        from logging_config import setup_logging
        base_dir = setup_logging()
        
        # Set up data directories
        data_dir = base_dir / 'data'
        self.failed_messages_path = data_dir / 'failed_messages'
        self.dlq_path = data_dir / 'dead_letter_queue'
        
        self.processing_queue = Queue()
        self.retry_counts = {}

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def start(self):
        """Start the retry processor."""
        if self.retry_thread is not None and self.retry_thread.is_alive():
            logger.warning("Retry processor is already running")
            return

        self.running = True
        self.retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
        self.retry_thread.start()
        logger.info("Retry processor started")

    def stop(self):
        """Stop the retry processor."""
        self.running = False
        if self.retry_thread is not None:
            self.retry_thread.join()
        logger.info("Retry processor stopped")

    def add_failed_message(self, topic: str, message: Dict[str, Any], error: str):
        """
        Add a failed message to the retry queue.
        
        Args:
            topic: The Kafka topic
            message: The message that failed to send
            error: The error message
        """
        try:
            timestamp = datetime.now().isoformat()
            message_id = f"{topic}-{timestamp}-{hash(str(message))}"
            
            failed_message = {
                'id': message_id,
                'topic': topic,
                'message': message,
                'error': error,
                'timestamp': timestamp,
                'retry_count': 0,
                'last_retry': None,
                'status': 'pending'
            }

            # Save to file
            file_path = self.failed_messages_path / f"{message_id}.json"
            with open(file_path, 'w') as f:
                json.dump(failed_message, f, indent=2)

            # Add to processing queue
            self.processing_queue.put(message_id)
            logger.info(f"Added failed message to retry queue: {message_id}")

        except Exception as e:
            logger.error(f"Error adding failed message to retry queue: {str(e)}")

    def _retry_loop(self):
        """Main retry processing loop."""
        while self.running:
            try:
                # Process messages from the queue
                while not self.processing_queue.empty():
                    message_id = self.processing_queue.get_nowait()
                    self._process_retry(message_id)
                    self.processing_queue.task_done()

                # Check for new failed messages on disk
                self._load_pending_messages()

                # Sleep before next iteration
                time.sleep(self.retry_interval)

            except Exception as e:
                logger.error(f"Error in retry loop: {str(e)}")
                time.sleep(10)  # Sleep briefly before continuing

    def _load_pending_messages(self):
        """Load pending messages from disk."""
        try:
            for file_path in self.failed_messages_path.glob("*.json"):
                message_id = file_path.stem
                if message_id not in self.retry_counts:
                    self.processing_queue.put(message_id)
                    self.retry_counts[message_id] = 0

        except Exception as e:
            logger.error(f"Error loading pending messages: {str(e)}")

    def _process_retry(self, message_id: str):
        """
        Process a single retry attempt.
        
        Args:
            message_id: The unique identifier for the failed message
        """
        try:
            file_path = self.failed_messages_path / f"{message_id}.json"
            if not file_path.exists():
                logger.warning(f"Message file not found: {message_id}")
                return

            # Load message data
            with open(file_path, 'r') as f:
                failed_message = json.load(f)

            # Check retry count
            if failed_message['retry_count'] >= self.max_retries:
                self._move_to_dead_letter_queue(failed_message)
                return

            # Attempt to resend
            self._attempt_resend(failed_message, file_path)

        except Exception as e:
            logger.error(f"Error processing retry for message {message_id}: {str(e)}")

    def _attempt_resend(self, failed_message: Dict[str, Any], file_path: Path):
        """
        Attempt to resend a failed message.
        
        Args:
            failed_message: The failed message data
            file_path: Path to the message file
        """
        try:
            # Attempt to send
            self.kafka_client._send_to_topic(
                failed_message['topic'],
                failed_message['message']
            )

            # If successful, remove the file
            file_path.unlink()
            if failed_message['id'] in self.retry_counts:
                del self.retry_counts[failed_message['id']]
            logger.info(f"Successfully resent message: {failed_message['id']}")

        except Exception as e:
            # Update retry count and timestamp
            failed_message['retry_count'] += 1
            failed_message['last_retry'] = datetime.now().isoformat()
            failed_message['last_error'] = str(e)

            # Save updated status
            with open(file_path, 'w') as f:
                json.dump(failed_message, f, indent=2)

            # Update retry count
            self.retry_counts[failed_message['id']] = failed_message['retry_count']

            logger.warning(
                f"Retry attempt {failed_message['retry_count']} failed for message "
                f"{failed_message['id']}: {str(e)}"
            )

    def _move_to_dead_letter_queue(self, failed_message: Dict[str, Any]):
        """
        Move a message to the dead letter queue after max retries.
        
        Args:
            failed_message: The failed message data
        """
        try:
            # Create dead letter queue directory if it doesn't exist
            dlq_path = Path('dead_letter_queue')
            dlq_path.mkdir(exist_ok=True)

            # Move message to DLQ
            message_id = failed_message['id']
            source_path = self.failed_messages_path / f"{message_id}.json"
            dest_path = dlq_path / f"{message_id}.json"

            # Update status and move file
            failed_message['status'] = 'dead_letter'
            failed_message['moved_to_dlq_at'] = datetime.now().isoformat()

            with open(dest_path, 'w') as f:
                json.dump(failed_message, f, indent=2)

            # Remove original file
            source_path.unlink()
            if message_id in self.retry_counts:
                del self.retry_counts[message_id]

            logger.info(f"Moved message to dead letter queue: {message_id}")

            # Notify Kafka about DLQ message
            self.kafka_client._send_to_dead_letter_queue(
                failed_message['message'],
                failed_message.get('last_error', 'Max retries exceeded')
            )

        except Exception as e:
            logger.error(f"Error moving message to dead letter queue: {str(e)}")

    def get_retry_status(self) -> Dict[str, Any]:
        """Get the current status of retry processing."""
        try:
            total_pending = len(list(self.failed_messages_path.glob("*.json")))
            dlq_path = Path('dead_letter_queue')
            total_dlq = len(list(dlq_path.glob("*.json"))) if dlq_path.exists() else 0

            return {
                'total_pending': total_pending,
                'total_dlq': total_dlq,
                'retry_counts': self.retry_counts.copy(),
                'queue_size': self.processing_queue.qsize(),
                'is_running': self.running
            }

        except Exception as e:
            logger.error(f"Error getting retry status: {str(e)}")
            return {}

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, stopping retry processor...")
        self.stop()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Kafka client and retry processor
    from besai_kafka_client import BeSAIKafkaClient
    
    kafka_client = BeSAIKafkaClient()
    retry_processor = RetryProcessor(kafka_client)
    
    try:
        # Start the retry processor
        retry_processor.start()
        
        # Simulate some failed messages
        retry_processor.add_failed_message(
            "knowledge_transfer",
            {"data": "test_message_1"},
            "Connection error"
        )
        
        retry_processor.add_failed_message(
            "cognitive_update",
            {"data": "test_message_2"},
            "Timeout error"
        )
        
        # Monitor status
        while True:
            status = retry_processor.get_retry_status()
            logger.info(f"Current retry status: {status}")
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        retry_processor.stop()
        kafka_client.close()
