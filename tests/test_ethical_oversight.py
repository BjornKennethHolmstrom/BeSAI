import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from src.besai.ethics.ethical_oversight_client import EthicalOversightClient
from src.besai.ethics.ethical_hooks import EthicalHooks, EthicalContext

class TestEthicalOversight(unittest.TestCase):
    def setUp(self):
        self.mock_client = Mock(spec=EthicalOversightClient)
        self.ethical_hooks = EthicalHooks(self.mock_client)

    def test_evaluate_knowledge_update(self):
        # Test data
        entity = "AI"
        attributes = {"type": "technology", "description": "Artificial Intelligence"}
        source = "user_input"
        certainty = 0.9

        # Mock the client response
        self.mock_client.evaluate_action.return_value.is_approved = True
        self.mock_client.evaluate_action.return_value.reasoning = "Approved"

        # Test the evaluation
        is_approved, reasoning = self.ethical_hooks.evaluate_knowledge_update(
            entity, attributes, source, certainty
        )

        # Assertions
        self.assertTrue(is_approved)
        self.assertEqual(reasoning, "Approved")
        self.mock_client.evaluate_action.assert_called_once()

    def test_evaluate_reasoning_output(self):
        # Test data
        query = "What is AI?"
        result = "AI is a field of computer science..."
        confidence = 0.8

        # Mock the client response
        self.mock_client.evaluate_action.return_value.is_approved = True
        self.mock_client.evaluate_action.return_value.reasoning = "Content approved"

        # Test the evaluation
        is_approved, reasoning = self.ethical_hooks.evaluate_reasoning_output(
            query, result, confidence
        )

        # Assertions
        self.assertTrue(is_approved)
        self.assertEqual(reasoning, "Content approved")
        self.mock_client.evaluate_action.assert_called_once()

    def test_evaluate_cognitive_update(self):
        # Test data
        cognitive_state = {
            "attention_focus": "AI ethics",
            "emotional_state": "neutral",
            "certainty": 0.9
        }
        nlp_analysis = {
            "entities": [{"text": "AI", "label": "TECHNOLOGY"}]
        }

        # Mock the client response
        self.mock_client.evaluate_action.return_value.is_approved = True
        self.mock_client.evaluate_action.return_value.reasoning = "Update approved"

        # Test the evaluation
        is_approved, reasoning = self.ethical_hooks.evaluate_cognitive_update(
            cognitive_state, nlp_analysis
        )

        # Assertions
        self.assertTrue(is_approved)
        self.assertEqual(reasoning, "Update approved")
        self.mock_client.evaluate_action.assert_called_once()

    def test_rejected_action(self):
        # Mock a rejected action
        self.mock_client.evaluate_action.return_value.is_approved = False
        self.mock_client.evaluate_action.return_value.reasoning = "Ethical violation detected"

        # Test with potentially problematic content
        is_approved, reasoning = self.ethical_hooks.evaluate_reasoning_output(
            "query", "potentially problematic content", 0.5
        )

        # Assertions
        self.assertFalse(is_approved)
        self.assertEqual(reasoning, "Ethical violation detected")

    def test_error_handling(self):
        # Mock an error in the client
        self.mock_client.evaluate_action.side_effect = Exception("Service error")

        # Test error handling
        is_approved, reasoning = self.ethical_hooks.evaluate_reasoning_output(
            "query", "content", 0.5
        )

        # Assertions
        self.assertFalse(is_approved)  # Should fail closed
        self.assertIn("Error during ethical evaluation", reasoning)

if __name__ == '__main__':
    unittest.main()
