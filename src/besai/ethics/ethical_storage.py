import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class EthicalDecision:
    decision_id: str
    timestamp: str
    action_type: str
    description: str
    context: Dict[str, str]
    is_approved: bool
    confidence: float
    reasoning: str
    violated_guidelines: List[str]
    suggested_modifications: List[str]
    metadata: Dict[str, str]

class EthicalStorage:
    def __init__(self, data_dir: Path):
        """
        Initialize the ethical storage system.
        
        Args:
            data_dir: Directory for storing ethical data
        """
        self.data_dir = data_dir / 'ethical_data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / 'ethical_decisions.db'
        self.cache_path = self.data_dir / 'guidelines_cache.json'
        
        self.lock = threading.Lock()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the SQLite database."""
        with self._get_db() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS ethical_decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    context TEXT NOT NULL,
                    is_approved INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    violated_guidelines TEXT NOT NULL,
                    suggested_modifications TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON ethical_decisions(timestamp)
            """)
            
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_action_type 
                ON ethical_decisions(action_type)
            """)

    @contextmanager
    def _get_db(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn.cursor()
            conn.commit()
        finally:
            conn.close()

    def store_decision(self, decision: EthicalDecision):
        """Store an ethical decision."""
        try:
            with self.lock, self._get_db() as db:
                db.execute("""
                    INSERT INTO ethical_decisions (
                        decision_id, timestamp, action_type, description, 
                        context, is_approved, confidence, reasoning,
                        violated_guidelines, suggested_modifications, 
                        metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.decision_id,
                    decision.timestamp,
                    decision.action_type,
                    decision.description,
                    json.dumps(decision.context),
                    1 if decision.is_approved else 0,
                    decision.confidence,
                    decision.reasoning,
                    json.dumps(decision.violated_guidelines),
                    json.dumps(decision.suggested_modifications),
                    json.dumps(decision.metadata),
                    datetime.now().isoformat()
                ))
                logger.info(f"Stored ethical decision: {decision.decision_id}")
        
        except Exception as e:
            logger.error(f"Error storing ethical decision: {str(e)}")
            raise

    def get_decision(self, decision_id: str) -> Optional[EthicalDecision]:
        """Retrieve a specific ethical decision."""
        try:
            with self.lock, self._get_db() as db:
                db.execute("""
                    SELECT * FROM ethical_decisions 
                    WHERE decision_id = ?
                """, (decision_id,))
                
                row = db.fetchone()
                if row:
                    return self._row_to_decision(row)
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving ethical decision: {str(e)}")
            return None

    def get_decisions_by_type(self, action_type: str, limit: int = 100) -> List[EthicalDecision]:
        """Retrieve ethical decisions by action type."""
        try:
            with self.lock, self._get_db() as db:
                db.execute("""
                    SELECT * FROM ethical_decisions 
                    WHERE action_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (action_type, limit))
                
                return [self._row_to_decision(row) for row in db.fetchall()]
                
        except Exception as e:
            logger.error(f"Error retrieving decisions by type: {str(e)}")
            return []

    def get_recent_decisions(self, limit: int = 100) -> List[EthicalDecision]:
        """Retrieve recent ethical decisions."""
        try:
            with self.lock, self._get_db() as db:
                db.execute("""
                    SELECT * FROM ethical_decisions 
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                return [self._row_to_decision(row) for row in db.fetchall()]
                
        except Exception as e:
            logger.error(f"Error retrieving recent decisions: {str(e)}")
            return []

    def _row_to_decision(self, row) -> EthicalDecision:
        """Convert a database row to an EthicalDecision object."""
        return EthicalDecision(
            decision_id=row[0],
            timestamp=row[1],
            action_type=row[2],
            description=row[3],
            context=json.loads(row[4]),
            is_approved=bool(row[5]),
            confidence=row[6],
            reasoning=row[7],
            violated_guidelines=json.loads(row[8]),
            suggested_modifications=json.loads(row[9]),
            metadata=json.loads(row[10])
        )

    def cache_guidelines(self, guidelines: Dict[str, Any]):
        """Cache ethical guidelines."""
        try:
            with self.lock:
                guidelines['cached_at'] = datetime.now().isoformat()
                self.cache_path.write_text(json.dumps(guidelines, indent=2))
                logger.info("Updated guidelines cache")
                
        except Exception as e:
            logger.error(f"Error caching guidelines: {str(e)}")

    def get_cached_guidelines(self) -> Optional[Dict[str, Any]]:
        """Retrieve cached guidelines."""
        try:
            if not self.cache_path.exists():
                return None
                
            with self.lock:
                guidelines = json.loads(self.cache_path.read_text())
                cached_at = datetime.fromisoformat(guidelines['cached_at'])
                
                # Check if cache is still valid (less than 1 hour old)
                if (datetime.now() - cached_at).total_seconds() < 3600:
                    return guidelines
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cached guidelines: {str(e)}")
            return None

    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about ethical decisions."""
        try:
            with self.lock, self._get_db() as db:
                # Total decisions
                db.execute("SELECT COUNT(*) FROM ethical_decisions")
                total_decisions = db.fetchone()[0]
                
                # Approval rate
                db.execute("""
                    SELECT 
                        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ethical_decisions)
                    FROM ethical_decisions 
                    WHERE is_approved = 1
                """)
                approval_rate = db.fetchone()[0]
                
                # Decisions by type
                db.execute("""
                    SELECT action_type, COUNT(*) 
                    FROM ethical_decisions 
                    GROUP BY action_type
                """)
                decisions_by_type = dict(db.fetchall())
                
                # Average confidence
                db.execute("""
                    SELECT AVG(confidence) 
                    FROM ethical_decisions
                """)
                avg_confidence = db.fetchone()[0]
                
                return {
                    "total_decisions": total_decisions,
                    "approval_rate": approval_rate,
                    "decisions_by_type": decisions_by_type,
                    "average_confidence": avg_confidence,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error generating decision statistics: {str(e)}")
            return {}

    def cleanup_old_decisions(self, days_to_keep: int = 30):
        """Clean up old ethical decisions."""
        try:
            cleanup_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            with self.lock, self._get_db() as db:
                db.execute("""
                    DELETE FROM ethical_decisions 
                    WHERE timestamp < ?
                """, (cleanup_date,))
                
                deleted_count = db.rowcount
                logger.info(f"Cleaned up {deleted_count} old ethical decisions")
                
        except Exception as e:
            logger.error(f"Error cleaning up old decisions: {str(e)}")
