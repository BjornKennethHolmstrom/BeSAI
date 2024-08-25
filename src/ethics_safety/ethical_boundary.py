# BeSAI/src/ethics_safety/ethical_boundary.py

import os
import pickle
import abc
from typing import Any, List, Dict
from enum import Enum
from ethical_learning_model import EthicalLearningModel
from detailed_context_generator import DetailedContextGenerator
from what_if_analysis import WhatIfAnalysis, print_analysis_results
from ethical_types import EthicalEvaluator
from logger import logger

class EthicalPrinciple(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, action: Any, context: Dict[str, Any]) -> float:
        pass

    @abc.abstractmethod
    def explain(self, action: Any, context: Dict[str, Any]) -> str:
        pass

class UtilitarianPrinciple(EthicalPrinciple):
    def __init__(self):
        self.impact_weights = {
            'short_term': 0.3,
            'medium_term': 0.3,
            'long_term': 0.4
        }

    def evaluate(self, action: Any, context: Dict[str, Any]) -> float:
        impacts = context.get('impacts', {})
        weighted_score = sum(
            self.impact_weights[term] * (impacts.get(term, {}).get('positive', 0) - 
                                         impacts.get(term, {}).get('negative', 0))
            for term in self.impact_weights
        )
        return (weighted_score + 10) / 20  # Normalize to 0-1 range

    def explain(self, action: Any, context: Dict[str, Any]) -> str:
        score = self.evaluate(action, context)
        impacts = context.get('impacts', {})
        explanation = f"Utilitarian principle score: {score:.2f}\n"
        for term in self.impact_weights:
            pos = impacts.get(term, {}).get('positive', 0)
            neg = impacts.get(term, {}).get('negative', 0)
            explanation += f"  {term.capitalize()} term impact: +{pos} / -{neg}\n"
        return explanation

class DeontologicalPrinciple(EthicalPrinciple):
    def __init__(self):
        self.ethical_rules = [
            ('no_lying', self.check_no_lying),
            ('privacy_respect', self.check_privacy_respect),
            ('no_harm', self.check_no_harm)
        ]

    def check_no_lying(self, context):
        return not context.get('involves_lying', False)

    def check_privacy_respect(self, context):
        return not context.get('violates_privacy', False)

    def check_no_harm(self, context):
        return not context.get('causes_harm', False)

    def evaluate(self, action: Any, context: Dict[str, Any]) -> float:
        return sum(rule[1](context) for rule in self.ethical_rules) / len(self.ethical_rules)

    def explain(self, action: Any, context: Dict[str, Any]) -> str:
        explanation = f"Deontological principle score: {self.evaluate(action, context):.2f}\n"
        for rule_name, rule_func in self.ethical_rules:
            passes = rule_func(context)
            explanation += f"  {rule_name.replace('_', ' ').capitalize()}: {'Passes' if passes else 'Fails'}\n"
        return explanation

class VirtueEthicsPrinciple(EthicalPrinciple):
    def __init__(self):
        self.virtues = {
            'honesty': {'weight': 0.25, 'threshold': 7},
            'compassion': {'weight': 0.25, 'threshold': 6},
            'courage': {'weight': 0.2, 'threshold': 5},
            'wisdom': {'weight': 0.3, 'threshold': 8}
        }

    def evaluate(self, action: Any, context: Dict[str, Any]) -> float:
        return sum(
            self.virtues[virtue]['weight'] * (context.get(f'{virtue}_score', 0) / 10)
            for virtue in self.virtues
        )

    def explain(self, action: Any, context: Dict[str, Any]) -> str:
        score = self.evaluate(action, context)
        explanation = f"Virtue Ethics principle score: {score:.2f}\n"
        for virtue, details in self.virtues.items():
            virtue_score = context.get(f'{virtue}_score', 0)
            explanation += f"  {virtue.capitalize()}: {virtue_score}/10 "
            explanation += f"({'Above' if virtue_score > details['threshold'] else 'Below'} threshold)\n"
        return explanation

class HarmLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    SEVERE = 4

class HarmMinimizationPrinciple(EthicalPrinciple):
    def __init__(self):
        self.harm_weights = {
            'physical': 0.4,
            'emotional': 0.3,
            'financial': 0.2,
            'social': 0.1
        }

    def evaluate(self, action: Any, context: Dict[str, Any]) -> float:
        harm_scores = context.get('harm_scores', {})
        weighted_harm = sum(
            self.harm_weights[harm_type] * HarmLevel[level.upper()].value
            for harm_type, level in harm_scores.items()
        )
        return 1 - (weighted_harm / (4 * sum(self.harm_weights.values())))  # Normalize to 0-1 range

    def explain(self, action: Any, context: Dict[str, Any]) -> str:
        score = self.evaluate(action, context)
        harm_scores = context.get('harm_scores', {})
        explanation = f"Harm Minimization principle score: {score:.2f}\n"
        for harm_type, level in harm_scores.items():
            explanation += f"  {harm_type.capitalize()} harm: {level}\n"
        return explanation

class FairnessPrinciple(EthicalPrinciple):
    def __init__(self):
        self.fairness_aspects = {
            'equal_opportunity': 0.3,
            'proportional_outcome': 0.3,
            'lack_of_bias': 0.4
        }

    def evaluate(self, action: Any, context: Dict[str, Any]) -> float:
        fairness_scores = context.get('fairness_scores', {})
        return sum(
            self.fairness_aspects[aspect] * (fairness_scores.get(aspect, 0) / 10)
            for aspect in self.fairness_aspects
        )

    def explain(self, action: Any, context: Dict[str, Any]) -> str:
        score = self.evaluate(action, context)
        fairness_scores = context.get('fairness_scores', {})
        explanation = f"Fairness principle score: {score:.2f}\n"
        for aspect in self.fairness_aspects:
            aspect_score = fairness_scores.get(aspect, 0)
            explanation += f"  {aspect.replace('_', ' ').capitalize()}: {aspect_score}/10\n"
        return explanation

class EthicalBoundary:
    def __init__(self, ethical_threshold: float = 0.5):
        self.principles: List[EthicalPrinciple] = [
            UtilitarianPrinciple(),
            DeontologicalPrinciple(),
            VirtueEthicsPrinciple(),
            HarmMinimizationPrinciple(),
            FairnessPrinciple()
        ]
        self.learning_model = EthicalLearningModel(num_principles=len(self.principles))
        self.context_generator = DetailedContextGenerator()
        self.ethical_threshold = ethical_threshold

        logger.info("Initialized EthicalBoundary")

    def set_ethical_threshold(self, new_threshold: float):
        """
        Set a new ethical threshold.
        
        Args:
            new_threshold (float): The new threshold value between 0 and 1.
        """
        if 0 <= new_threshold <= 1:
            self.ethical_threshold = new_threshold
            logger.info(f"Ethical threshold updated to {new_threshold}")
        else:
            logger.warning(f"Invalid ethical threshold {new_threshold}. Must be between 0 and 1.")

    def is_action_ethical(self, action: str, context: Dict[str, Any]) -> bool:
        score = self.get_ethical_score(action, context)
        return score >= self.ethical_threshold

    def get_ethical_explanation(self, action: str, context: Dict[str, Any]) -> str:
        explanations = [principle.explain(action, context) for principle in self.principles]
        principle_scores = [principle.evaluate(action, context) for principle in self.principles]
        overall_score = self.learning_model.predict(principle_scores)
        overall_decision = "ETHICAL" if self.is_action_ethical(action, context) else "UNETHICAL"
        
        explanation = f"Action: {action}\n\n"
        explanation += "Context:\n"
        for key, value in context.items():
            if key != 'action':  # Skip 'action' as it's already displayed
                explanation += f"  {key}: {value}\n"
        explanation += f"\nOverall decision: Action is {overall_decision} (Score: {overall_score:.2f})\n\n"
        explanation += "Principle Scores:\n"
        importances = self.learning_model.get_principle_importances()
        for i, (principle, score, importance) in enumerate(zip(self.principles, principle_scores, importances)):
            explanation += f"  {principle.__class__.__name__}: {score:.2f} (Importance: {importance:.4f})\n"
        explanation += "\nDetailed Explanations:\n" + "\n".join(explanations)
        return explanation

    def get_ethical_score(self, action: str, context: Dict[str, Any]) -> float:
        principle_scores = [principle.evaluate(action, context) for principle in self.principles]
        return self.learning_model.predict(principle_scores)

    def provide_feedback(self, action: str, context: Dict[str, Any], user_feedback: float, force_update: bool = False):
        principle_scores = [principle.evaluate(action, context) for principle in self.principles]
        self.learning_model.update(principle_scores, user_feedback, force_update)
        logger.info(f"Feedback provided for action '{action}': {user_feedback}")

    def get_learning_summary(self) -> str:
        return self.learning_model.get_learning_summary()

    def save_system(self, filepath: str):
        filepath = os.path.expanduser(filepath)
        self.learning_model.save_model(f"{filepath}_learning_model")
        
        # Save principles and context generator separately
        with open(f"{filepath}_principles.pkl", 'wb') as f:
            pickle.dump(self.principles, f)
        
        with open(f"{filepath}_context_generator.pkl", 'wb') as f:
            pickle.dump(self.context_generator, f)
        logger.info(f"Ethical system saved to {filepath}")

    @classmethod
    def load_system(cls, filepath: str):
        filepath = os.path.expanduser(filepath)
        instance = cls()
        
        # Check if all required files exist
        required_files = [
            f"{filepath}_learning_model_keras_model.keras",
            f"{filepath}_learning_model_attributes.pkl",
            f"{filepath}_principles.pkl",
            f"{filepath}_context_generator.pkl"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"The following required files are missing: {', '.join(missing_files)}")
        
        # If all files exist, proceed with loading
        instance.learning_model = EthicalLearningModel.load_model(f"{filepath}_learning_model")
        
        with open(f"{filepath}_principles.pkl", 'rb') as f:
            instance.principles = pickle.load(f)
        
        with open(f"{filepath}_context_generator.pkl", 'rb') as f:
            instance.context_generator = pickle.load(f)

        logger.info(f"Ethical system loaded from {filepath}")        
        return instance

# Usage example
ethical_boundary = EthicalBoundary()

def check_action_ethics(action: str, context: Dict[str, Any]):
    if ethical_boundary.is_action_ethical(action, context):
        print("Action is ethical")
    else:
        print("Action is not ethical")
    print(ethical_boundary.get_ethical_explanation(action, context))

def train_ethical_boundary():
    while True:
        context = ethical_boundary.context_generator.generate_context()
        action = context['action']
        check_action_ethics(action, context)
        user_feedback = float(input("Enter your ethical score for this action (0-1): "))
        
        force_update = False
        if user_feedback < ethical_boundary.learning_model.ethical_threshold:
            force_update = input("This feedback is below the ethical threshold. Do you want to force the update? (y/n): ").lower() == 'y'
        
        # Provide more detailed feedback about the learning process
        previous_weights = ethical_boundary.learning_model.get_principle_importances()
        ethical_boundary.provide_feedback(action, context, user_feedback, force_update)
        current_weights = ethical_boundary.learning_model.get_principle_importances()
        
        print("\nLearning Update:")
        for i, (prev, curr) in enumerate(zip(previous_weights, current_weights)):
            principle_name = ethical_boundary.principles[i].__class__.__name__
            change = curr - prev
            print(f"{principle_name}: {prev:.4f} -> {curr:.4f} (Change: {change:+.4f})")
        
        print("\nLearning Summary:")
        print(ethical_boundary.get_learning_summary())

        logger.debug(f"Training iteration completed. User feedback: {user_feedback}")        
        if input("\nContinue training? (y/n): ").lower() != 'y':
            break
    logger.info("Ethical boundary training completed")

def perform_what_if_analysis(ethical_boundary: EthicalBoundary):
    from what_if_analysis import WhatIfAnalysis, print_analysis_results
    
    what_if = WhatIfAnalysis(ethical_boundary)
    
    action = input("Enter the action to analyze: ")
    context = {}
    print("Enter the original context (enter an empty key to finish):")
    while True:
        key = input("Enter context key: ")
        if not key:
            break
        value = input(f"Enter value for {key}: ")
        context[key] = value
    
    changes = []
    print("Enter the changes to analyze (enter an empty key to finish):")
    while True:
        key = input("Enter change key: ")
        if not key:
            break
        value = input(f"Enter new value for {key}: ")
        changes.append((key, value))
    
    results = what_if.analyze_changes(action, context, changes)
    print_analysis_results(results)

def save_model(filepath: str):
    ethical_boundary.save_system(filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: str):
    global ethical_boundary
    try:
        ethical_boundary = EthicalBoundary.load_system(filepath)
        print(f"Model successfully loaded from {filepath}")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please check the filepath and ensure all required files are present.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {str(e)}")
        print("Please check the filepath and ensure all files are not corrupted.")

def main():
    logger.info("BeSAI ethical boundary script started")
    
    ethical_boundary = EthicalBoundary(ethical_threshold=0.5)  # You can set the initial threshold here
    
    while True:
        print("\n1. Train model")
        print("2. Save model")
        print("3. Load model")
        print("4. Perform What-If Analysis")
        print("5. Set Ethical Threshold")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            train_ethical_boundary()
        elif choice == '2':
            filepath = input("Enter filepath to save the model: ")
            ethical_boundary.save_system(filepath)
        elif choice == '3':
            filepath = input("Enter filepath to load the model: ")
            ethical_boundary = EthicalBoundary.load_system(filepath)
        elif choice == '4':
            perform_what_if_analysis(ethical_boundary)
        elif choice == '5':
            new_threshold = float(input("Enter new ethical threshold (0-1): "))
            ethical_boundary.set_ethical_threshold(new_threshold)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")
    
    logger.info("BeSAI ethical boundary script completed")

if __name__ == "__main__":
    main()
