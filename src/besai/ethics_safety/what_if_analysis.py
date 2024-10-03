# BeSAI/src/ethics_safety/what_if_analysis.py

from typing import Dict, Any, List, Tuple
from .ethical_types import EthicalEvaluator
from .logger import logger

class WhatIfAnalysis:
    def __init__(self, ethical_evaluator: EthicalEvaluator):
        self.ethical_evaluator = ethical_evaluator

    def analyze_changes(self, action: str, original_context: Dict[str, Any], changes: List[Tuple[str, Any]]) -> Dict[str, Any]:
        logger.info(f"Performing what-if analysis for action: {action}")
        
        original_ethical = self.ethical_evaluator.is_action_ethical(action, original_context)
        original_score = self.ethical_evaluator.get_ethical_score(action, original_context)
        
        results = {
            "original": {
                "ethical": original_ethical,
                "score": original_score
            },
            "changes": []
        }

        for change_key, change_value in changes:
            modified_context = original_context.copy()
            self._apply_change(modified_context, change_key, change_value)
            
            modified_ethical = self.ethical_evaluator.is_action_ethical(action, modified_context)
            modified_score = self.ethical_evaluator.get_ethical_score(action, modified_context)
            
            change_result = {
                "change": f"{change_key} -> {change_value}",
                "ethical": modified_ethical,
                "score": modified_score,
                "score_difference": modified_score - original_score
            }
            results["changes"].append(change_result)
            
            logger.info(f"What-if change applied: {change_key} -> {change_value}")
            logger.info(f"Result: Ethical: {modified_ethical}, Score: {modified_score}")

        return results

    def _apply_change(self, context: Dict[str, Any], key: str, value: Any):
        keys = key.split('.')
        current = context
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

def print_analysis_results(results: Dict[str, Any]):
    print("\nWhat-If Analysis Results:")
    print(f"Original scenario - Ethical: {results['original']['ethical']}, Score: {results['original']['score']:.4f}")
    print("\nScenario Changes:")
    for change in results['changes']:
        print(f"- Change: {change['change']}")
        print(f"  Ethical: {change['ethical']}, Score: {change['score']:.4f}, Difference: {change['score_difference']:+.4f}")
        print()

# Example usage
if __name__ == "__main__":
    print("This module is not meant to be run directly. Import and use the WhatIfAnalysis class in your main script.")
