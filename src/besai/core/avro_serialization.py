import unittest
from unittest.mock import patch, MagicMock
import json
import os
from datetime import datetime
from besai.core.avro_serialization import AvroSerializer

class TestAvroSerializer(unittest.TestCase):
    def setUp(self):
        self.serializer = AvroSerializer()
        self.maxDiff = None

    def test_knowledge_transfer_serialization(self):
        # Test data
        knowledge_graph = {
            "entities": [
                {
                    "text": "AI",
                    "label": "TECHNOLOGY",
                    "attributes": {"description": "Artificial Intelligence"},
                    "certainty": 0.9
                }
            ],
            "relationships": [
                {
                    "subject": "AI",
                    "predicate": "is_type_of",
                    "object": "Technology",
                    "attributes": {"confidence": "high"},
                    "certainty": 0.8
                }
            ]
        }

        # Serialize and deserialize
        avro_bytes = self.serializer.serialize_knowledge_transfer(knowledge_graph)
        result = self.serializer.deserialize_knowledge_transfer(avro_bytes)

        # Check entities
        self.assertEqual(len(result["entities"]), 1)
        entity = result["entities"][0]
        self.assertEqual(entity["text"], "AI")
        self.assertEqual(entity["label"], "TECHNOLOGY")
        self.assertEqual(entity["attributes"]["description"], "Artificial Intelligence")
        self.assertEqual(entity["certainty"], 0.9)

        # Check relationships
        self.assertEqual(len(result["relationships"]), 1)
        rel = result["relationships"][0]
        self.assertEqual(rel["subject"], "AI")
        self.assertEqual(rel["predicate"], "is_type_of")
        self.assertEqual(rel["object"], "Technology")
        self.assertEqual(rel["attributes"]["confidence"], "high")
        self.assertEqual(rel["certainty"], 0.8)

    def test_cognitive_update_serialization(self):
        # Test data
        cognitive_state = {
            "attention_focus": "AI ethics",
            "emotional_state": "curious",
            "reasoning_depth": 3,
            "state_params": {
                "creativity_level": 0.7,
                "perception_shift": 0.6,
                "focus_level": 0.8
            },
            "focus_level": 0.9,
            "associative_thinking": 0.7
        }

        # Serialize and deserialize
        avro_bytes = self.serializer.serialize_cognitive_update(cognitive_state)
        result = self.serializer.deserialize_cognitive_update(avro_bytes)

        # Check fields
        self.assertEqual(result["attention_focus"], "AI ethics")
        self.assertEqual(result["emotional_state"], "curious")
        self.assertEqual(result["reasoning_depth"], 3)
        self.assertEqual(result["state_params"]["creativity_level"], 0.7)
        self.assertEqual(result["state_params"]["perception_shift"], 0.6)
        self.assertEqual(result["state_params"]["focus_level"], 0.8)
        self.assertEqual(result["focus_level"], 0.9)
        self.assertEqual(result["associative_thinking"], 0.7)

    def test_error_handling(self):
        # Test invalid knowledge graph
        with self.assertRaises(Exception):
            self.serializer.serialize_knowledge_transfer({"invalid": "data"})

        # Test invalid cognitive state
        with self.assertRaises(Exception):
            self.serializer.serialize_cognitive_update({"invalid": "data"})

        # Test invalid Avro bytes
        with self.assertRaises(Exception):
            self.serializer.deserialize_knowledge_transfer(b"invalid data")

        with self.assertRaises(Exception):
            self.serializer.deserialize_cognitive_update(b"invalid data")

if __name__ == '__main__':
    unittest.main()
