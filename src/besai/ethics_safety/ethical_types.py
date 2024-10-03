# BeSAI/src/ethics_safety/ethical_types.py

from typing import Protocol, Dict, Any

class EthicalEvaluator(Protocol):
    def is_action_ethical(self, action: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate whether an action is ethical given the context.
        
        Args:
            action (str): The action to evaluate.
            context (Dict[str, Any]): The context in which the action is performed.
        
        Returns:
            bool: True if the action is ethical, False otherwise.
        """
        pass

    def get_ethical_score(self, action: str, context: Dict[str, Any]) -> float:
        """
        Get a numerical ethical score for an action given the context.
        
        Args:
            action (str): The action to evaluate.
            context (Dict[str, Any]): The context in which the action is performed.
        
        Returns:
            float: A numerical score representing the ethical evaluation of the action.
        """
        pass
