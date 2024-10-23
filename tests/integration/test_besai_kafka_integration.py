import unittest
from unittest.mock import MagicMock, patch
from besai.integration.besai_kafka_client import BeSAIKafkaClient

class TestBeSAIKafkaIntegration(unittest.TestCase):

    @patch('besai_kafka_client.get_producer')
    @patch('besai_kafka_client.get_consumer')
    def setUp(self, mock_get_consumer, mock_get_producer):
        self.mock_producer = MagicMock()
        self.mock_consumer = MagicMock()
        mock_get_producer.return_value = self.mock_producer
        mock_get_consumer.return_value = self.mock_consumer
        self.kafka_client = BeSAIKafkaClient()

    def test_send_knowledge_transfer(self):
        knowledge_graph = {"entity": "AI", "relation": "is a subset of", "entity2": "Computer Science"}
        self.kafka_client.send_knowledge_transfer(knowledge_graph)
        self.mock_producer.send.assert_called_once()
        args, kwargs = self.mock_producer.send.call_args
        self.assertEqual(args[0], 'besai-knowledge-transfer')
        self.assertEqual(kwargs['value']['type'], 'knowledge_transfer')
        self.assertEqual(kwargs['value']['data'], knowledge_graph)

    def test_send_cognitive_update(self):
        cognitive_state = {"attention_focus": "ethics in AI", "emotional_state": "curious"}
        self.kafka_client.send_cognitive_update(cognitive_state)
        self.mock_producer.send.assert_called_once()
        args, kwargs = self.mock_producer.send.call_args
        self.assertEqual(args[0], 'besai-cognitive-update')
        self.assertEqual(kwargs['value']['type'], 'cognitive_update')
        self.assertEqual(kwargs['value']['data'], cognitive_state)

    def test_consume_quinca_updates(self):
        mock_callback = MagicMock()
        self.mock_consumer.__iter__.return_value = [
            MagicMock(value={"type": "cognitive_update", "data": {"attention_focus": "AI ethics"}}),
            MagicMock(value={"type": "knowledge_transfer", "data": {"entity": "AI", "relation": "impacts", "entity2": "society"}}),
        ]
        self.kafka_client.consume_quinca_updates(mock_callback)
        self.assertEqual(mock_callback.call_count, 2)

    def tearDown(self):
        self.kafka_client.close()

if __name__ == '__main__':
    unittest.main()
