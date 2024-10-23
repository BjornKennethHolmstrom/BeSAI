import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class EthicalContext:
    """Context information for ethical evaluation."""
    action_type: str
    description: str
    certainty: float
    context: Dict[str, str]
    timestamp: datetime
    source_component: str

class EthicalHooks:
    def __init__(self, ethical_client):
        """
        Initialize ethical hooks.
        
        Args:
            ethical_client: Instance of EthicalOversightClient
        """
        self.client = ethical_client
        self._initialize_guidelines()

    def _initialize_guidelines(self):
        """Initialize ethical guidelines."""
        try:
            self.guidelines = self.client.get_ethical_guidelines()
            logger.info("Initialized ethical guidelines")
        except Exception as e:
            logger.error(f"Failed to initialize ethical guidelines: {e}")
            self.guidelines = None

    def evaluate_knowledge_update(self, 
                                entity: str, 
                                attributes: Dict[str, Any], 
                                source: str, 
                                certainty: float) -> Tuple[bool, str]:
        """
        Evaluate a knowledge base update for ethical compliance.
        
        Returns:
            Tuple of (is_approved, reasoning)
        """
        context = EthicalContext(
            action_type="KNOWLEDGE_UPDATE",
            description=f"Update entity '{entity}' with attributes {attributes}",
            certainty=certainty,
            context={
                "source": source,
                "entity_type": attributes.get("type", "unknown"),
                "update_type": "modification" if entity in self.knowledge_base else "addition"
            },
            timestamp=datetime.now(),
            source_component="knowledge_base"
        )
        
        return self._evaluate_context(context)

    def evaluate_reasoning_output(self, 
                                query: str, 
                                result: str, 
                                confidence: float) -> Tuple[bool, str]:
        """
        Evaluate reasoning engine output for ethical compliance.
        
        Returns:
            Tuple of (is_approved, reasoning)
        """
        context = EthicalContext(
            action_type="REASONING_OUTPUT",
            description=result,
            certainty=confidence,
            context={
                "query": query,
                "response_length": str(len(result)),
                "contains_entities": str(any(entity in result for entity in self._get_sensitive_entities()))
            },
            timestamp=datetime.now(),
            source_component="reasoning_engine"
        )
        
        return self._evaluate_context(context)

    def evaluate_cognitive_update(self, 
                                cognitive_state: Dict[str, Any], 
                                nlp_analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate cognitive state updates for ethical compliance.
        
        Returns:
            Tuple of (is_approved, reasoning)
        """
        context = EthicalContext(
            action_type="COGNITIVE_UPDATE",
            description=f"Update cognitive state with {cognitive_state}",
            certainty=cognitive_state.get("certainty", 1.0),
            context={
                "emotional_state": cognitive_state.get("emotional_state", "neutral"),
                "attention_focus": cognitive_state.get("attention_focus", ""),
                "entity_types": str([e.get("label") for e in nlp_analysis.get("entities", [])])
            },
            timestamp=datetime.now(),
            source_component="cognitive_system"
        )
        
        return self._evaluate_context(context)

    def _evaluate_context(self, context: EthicalContext) -> Tuple[bool, str]:
        """
        Evaluate an ethical context and return the decision.
        
        Returns:
            Tuple of (is_approved, reasoning)
        """
        try:
            assessment = self.client.evaluate_action(
                action_type=context.action_type,
                description=context.description,
                context=context.context,
                certainty=context.certainty,
                metadata={
                    "timestamp": context.timestamp.isoformat(),
                    "source_component": context.source_component
                }
            )
            
            return assessment.is_approved, assessment.reasoning
        
        except Exception as e:
            logger.error(f"Error during ethical evaluation: {e}")
            # Conservative default: reject with explanation
            return False, f"Error during ethical evaluation: {str(e)}"

    def _get_sensitive_entities(self) -> List[str]:
        """Get list of sensitive entities from guidelines."""
        if not self.guidelines:
            return []
        
        sensitive_entities = []
        for guideline in self.guidelines.guidelines:
            if "sensitive_entity" in guideline.description.lower():
                sensitive_entities.extend(guideline.examples)
        
        return sensitive_entities
