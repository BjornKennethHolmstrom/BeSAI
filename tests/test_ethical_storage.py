import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import json
import sqlite3

from ethics.ethical_storage import EthicalStorage, EthicalDecision

class TestEthicalStorage(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.storage = EthicalStorage(Path(self.temp_dir))
        
        # Sample decision for testing
        self.sample_decision = EthicalDecision(
            decision_id="test-123",
            timestamp=datetime.now().isoformat(),
            action_type="TEST_ACTION",
            description="Test decision",
            context={"test": "context"},
            is_approved=True,
            confidence=0.9,
            reasoning="Test reasoning",
            violated_guidelines=[],
            suggested_modifications=[],
            metadata={"source": "test"}
        )

    def test_store_and_retrieve_decision(self):
        # Store decision
        self.storage.store_decision(self.sample_decision)
        
        # Retrieve decision
        retrieved = self.storage.get_decision(self.sample_decision.decision_id)
        
        # Verify
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.decision_id, self.sample_decision.decision_id)
        self.assertEqual(retrieved.action_type, self.sample_decision.action_type)
        self.assertEqual(retrieved.is_approved, self.sample_decision.is_approved)

    def test_get_decisions_by_type(self):
        # Store multiple decisions
        self.storage.store_decision(self.sample_decision)
        
        second_decision = EthicalDecision(
            decision_id="test-456",
            timestamp=datetime.now().isoformat(),
            action_type="ANOTHER_ACTION",
            description="Another test",
            context={},
            is_approved=False,
            confidence=0.8,
            reasoning="Another reasoning",
            violated_guidelines=["TEST_GUIDELINE"],
            suggested_modifications=[],
            metadata={}
        )
        self.storage.store_decision(second_decision)
        
        # Retrieve by type
        test_actions = self.storage.get_decisions_by_type("TEST_ACTION")
        another_actions = self.storage.get_decisions_by_type("ANOTHER_ACTION")
        
        # Verify
        self.assertEqual(len(test_actions), 1)
        self.assertEqual(len(another_actions), 1)
        self.assertEqual(test_actions[0].action_type, "TEST_ACTION")
        self.assertEqual(another_actions[0].action_type, "ANOTHER_ACTION")

    def test_get_recent_decisions(self):
        # Store decisions with different timestamps
        old_decision = EthicalDecision(
            decision_id="old-123",
            timestamp=(datetime.now() - timedelta(days=2)).isoformat(),
            action_type="OLD_ACTION",
            description="Old test",
            context={},
            is_approved=True,
            confidence=0.7,
            reasoning="Old reasoning",
            violated_guidelines=[],
            suggested_modifications=[],
            metadata={}
        )
        
        self.storage.store_decision(old_decision)
        self.storage.store_decision(self.sample_decision)
        
        # Retrieve recent decisions
        recent = self.storage.get_recent_decisions(limit=1)
        
        # Verify
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].decision_id, self.sample_decision.decision_id)

    def test_cache_guidelines(self):
        guidelines = {
            "version": "1.0",
            "guidelines": [
                {
                    "id": "G1",
                    "description": "Test guideline",
                    "importance": 0.8
                }
            ]
        }
        
        # Cache guidelines
        self.storage.cache_guidelines(guidelines)
        
        # Retrieve cached guidelines
        cached = self.storage.get_cached_guidelines()
        
        # Verify
        self.assertIsNotNone(cached)
        self.assertEqual(cached["version"], guidelines["version"])
        self.assertEqual(len(cached["guidelines"]), len(guidelines["guidelines"]))

    def test_cache_expiration(self):
        guidelines = {
            "version": "1.0",
            "guidelines": [],
            "cached_at": (datetime.now() - timedelta(hours=2)).isoformat()
        }
        
        # Cache outdated guidelines
        self.storage.cache_guidelines(guidelines)
        
        # Attempt to retrieve expired cache
        cached = self.storage.get_cached_guidelines()
        
        # Verify cache is expired
        self.assertIsNone(cached)

    def test_decision_statistics(self):
        # Store multiple decisions with different characteristics
        decisions = [
            EthicalDecision(
                decision_id=f"test-{i}",
                timestamp=datetime.now().isoformat(),
                action_type="TYPE_A" if i % 2 == 0 else "TYPE_B",
                description=f"Test {i}",
                context={},
                is_approved=i % 2 == 0,
                confidence=0.7 + (i / 10),
                reasoning=f"Reasoning {i}",
                violated_guidelines=[] if i % 2 == 0 else ["G1"],
                suggested_modifications=[],
                metadata={}
            ) for i in range(4)
        ]
        
        for decision in decisions:
            self.storage.store_decision(decision)
        
        # Get statistics
        stats = self.storage.get_decision_statistics()
        
        # Verify
        self.assertIn("total_decisions", stats)
        self.assertIn("approval_rate", stats)
        self.assertIn("decisions_by_type", stats)
        self.assertIn("average_confidence", stats)
        
        self.assertEqual(stats["total_decisions"], 4)
        self.assertGreater(stats["average_confidence"], 0)

    def test_cleanup_old_decisions(self):
        # Store old and new decisions
        old_decision = EthicalDecision(
            decision_id="old-123",
            timestamp=(datetime.now() - timedelta(days=31)).isoformat(),
            action_type="OLD_ACTION",
            description="Old test",
            context={},
            is_approved=True,
            confidence=0.7,
            reasoning="Old reasoning",
            violated_guidelines=[],
            suggested_modifications=[],
            metadata={}
        )
        
        self.storage.store_decision(old_decision)
        self.storage.store_decision(self.sample_decision)
        
        # Cleanup old decisions
        self.storage.cleanup_old_decisions(days_to_keep=30)
        
        # Verify only recent decision remains
        all_decisions = self.storage.get_recent_decisions()
        self.assertEqual(len(all_decisions), 1)
        self.assertEqual(all_decisions[0].decision_id, self.sample_decision.decision_id)

    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
