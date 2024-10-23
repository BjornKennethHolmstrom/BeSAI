import grpc
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from .ethical_oversight_pb2 import (
    ActionRequest, GuidelinesRequest, EthicalDecisionLog,
    EthicalAssessment, EthicalGuidelines
)
from .ethical_oversight_pb2_grpc import EthicalOversightStub

logger = logging.getLogger(__name__)

class EthicalOversightClient:
    def __init__(self, server_address: str = "localhost:50052", data_dir: Path = None):
        """
        Initialize the Ethical Oversight client.
        
        Args:
            server_address: The address of the ethical oversight service
            data_dir: Directory for storing ethical data (default: project_root/data)
        """
        self.channel = grpc.insecure_channel(server_address)
        self.stub = EthicalOversightStub(self.channel)
        
        if data_dir is None:
            data_dir = Path(os.path.dirname(os.path.dirname(__file__))) / 'data'
        
        self.storage = EthicalStorage(data_dir)
        self._guidelines_lock = threading.Lock()

    def evaluate_action(self, 
                       action_type: str,
                       description: str,
                       context: Dict[str, str] = None,
                       certainty: float = 1.0,
                       metadata: Dict[str, str] = None) -> EthicalAssessment:
        """Evaluate an action for ethical compliance with storage."""
        try:
            request = ActionRequest(
                action_type=action_type,
                description=description,
                context=context or {},
                certainty=certainty,
                metadata=metadata or {}
            )
            
            assessment = self.stub.EvaluateAction(request)
            
            # Store the decision
            decision = EthicalDecision(
                decision_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                action_type=action_type,
                description=description,
                context=context or {},
                is_approved=assessment.is_approved,
                confidence=assessment.confidence,
                reasoning=assessment.reasoning,
                violated_guidelines=list(assessment.violated_guidelines),
                suggested_modifications=list(assessment.suggested_modifications),
                metadata=metadata or {}
            )
            
            self.storage.store_decision(decision)
            return assessment
        
        except grpc.RpcError as e:
            logger.error(f"RPC error during ethical evaluation: {e}")
            # Check recent similar decisions
            similar_decisions = self.storage.get_decisions_by_type(action_type, limit=1)
            if similar_decisions:
                logger.info("Using most recent similar decision as fallback")
                last_decision = similar_decisions[0]
                return EthicalAssessment(
                    is_approved=last_decision.is_approved,
                    confidence=last_decision.confidence * 0.8,  # Reduce confidence for cached decision
                    reasoning=f"Using cached decision: {last_decision.reasoning}",
                    violated_guidelines=last_decision.violated_guidelines,
                    suggested_modifications=last_decision.suggested_modifications
                )
            
            return self._conservative_assessment()
        
        except Exception as e:
            logger.error(f"Error during ethical evaluation: {e}")
            return self._conservative_assessment()

    def get_ethical_guidelines(self, domain: str = "", force_refresh: bool = False) -> EthicalGuidelines:
        """Get ethical guidelines with caching."""
        try:
            if not force_refresh:
                cached = self.storage.get_cached_guidelines()
                if cached:
                    return EthicalGuidelines(**cached)

            with self._guidelines_lock:
                request = GuidelinesRequest(domain=domain)
                guidelines = self.stub.GetEthicalGuidelines(request)
                
                # Cache the guidelines
                self.storage.cache_guidelines({
                    'version': guidelines.version,
                    'last_updated': guidelines.last_updated,
                    'guidelines': [{
                        'id': g.id,
                        'description': g.description,
                        'rationale': g.rationale,
                        'importance': g.importance,
                        'examples': list(g.examples)
                    } for g in guidelines.guidelines]
                })
                
                return guidelines
        
        except Exception as e:
            logger.error(f"Error retrieving ethical guidelines: {e}")
            cached = self.storage.get_cached_guidelines()
            if cached:
                logger.info("Using cached guidelines due to error")
                return EthicalGuidelines(**cached)
            raise

    def _log_decision(self, action_request: ActionRequest, assessment: EthicalAssessment):
        """Log an ethical decision and its outcome."""
        try:
            log = EthicalDecisionLog(
                decision_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                action=action_request,
                assessment=assessment,
                final_outcome="ASSESSED",  # Initial outcome
                metadata={
                    "service_version": "1.0",
                    "client_id": "BeSAI"
                }
            )
            
            response = self.stub.LogEthicalDecision(log)
            if not response.success:
                logger.warning(f"Failed to log ethical decision: {response.message}")
        
        except Exception as e:
            logger.error(f"Error logging ethical decision: {e}")

    def _is_cache_valid(self) -> bool:
        """Check if the guidelines cache is still valid (less than 1 hour old)."""
        if self._cached_guidelines is None or self._guidelines_cache_time is None:
            return False
        
        cache_age = datetime.now() - self._guidelines_cache_time
        return cache_age.total_seconds() < 3600  # 1 hour cache validity

    def get_decision_history(self, limit: int = 100) -> List[EthicalDecision]:
        """Get recent ethical decisions."""
        return self.storage.get_recent_decisions(limit)

    def get_statistics(self) -> Dict[str, Any]:
        """Get ethical decision statistics."""
        return self.storage.get_decision_statistics()

    def _conservative_assessment(self) -> EthicalAssessment:
        """Create a conservative (rejecting) assessment for error cases."""
        return EthicalAssessment(
            is_approved=False,
            confidence=0.0,
            reasoning="Ethical oversight service unavailable - defaulting to conservative assessment",
            violated_guidelines=["SERVICE_UNAVAILABLE"],
            suggested_modifications=["Retry when service is available"]
        )

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old ethical decisions."""
        self.storage.cleanup_old_decisions(days_to_keep)

    def close(self):
        """Close the gRPC channel."""
        try:
            self.channel.close()
        except Exception as e:
            logger.error(f"Error closing ethical oversight client: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
