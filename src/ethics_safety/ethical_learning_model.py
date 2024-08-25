# BeSAI/src/ethics_safety/ethical_learning_model.py

import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import pickle
import os
from logger import logger

class CustomLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = initial_learning_rate
        self.decay_rate = 0.99

    def __call__(self, step):
        return self.current_learning_rate

    def decay(self):
        self.current_learning_rate *= self.decay_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "current_learning_rate": self.current_learning_rate,
            "decay_rate": self.decay_rate
        }

    @classmethod
    def from_config(cls, config):
        instance = cls(config['initial_learning_rate'])
        instance.current_learning_rate = config['current_learning_rate']
        instance.decay_rate = config['decay_rate']
        return instance

class EthicalLearningModel:
    def __init__(self, num_principles: int):
        self.num_principles = num_principles
        self.ethical_threshold = 0.3  # Minimum ethical score
        self.max_principle_weight = 0.5  # Maximum weight for any single principle
        self.initial_learning_rate = 0.01
        self.lr_schedule = CustomLearningRateSchedule(self.initial_learning_rate)
        self.history: List[Dict[str, Any]] = []
        self.model = self._build_model()
        logger.info(f"Initialized EthicalLearningModel with {num_principles} principles")

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.num_principles,)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=self.lr_schedule), loss='mse')
        return model

    def predict(self, principle_scores: List[float]) -> float:
        return self.model.predict(np.array([principle_scores]), verbose=0)[0][0]

    def update(self, principle_scores: List[float], user_feedback: float, force_update: bool = False):
        is_ethical = self._is_update_ethical(principle_scores, user_feedback)
        if is_ethical or force_update:
            self._perform_update(principle_scores, user_feedback)
            if not is_ethical:
                logger.warning("Update performed despite ethical concerns due to force_update flag.")
        else:
            logger.warning("Update rejected due to ethical safeguards. Use force_update=True to override.")

    def _perform_update(self, principle_scores: List[float], user_feedback: float):
        principle_scores = np.array([principle_scores])
        user_feedback = np.array([user_feedback])
        
        # Apply learning rate decay
        self.lr_schedule.decay()
        
        self.model.fit(principle_scores, user_feedback, verbose=0)
        
        predicted_score = self.predict(principle_scores[0])
        
        self.history.append({
            'principle_scores': principle_scores[0].tolist(),
            'predicted_score': predicted_score,
            'user_feedback': user_feedback[0],
            'learning_rate': self.lr_schedule.current_learning_rate,
        })
        logger.debug(f"Learning rate decayed to {self.lr_schedule.current_learning_rate}")

    def update_learning_rate(self):
        self.current_learning_rate *= self.learning_rate_decay
        # Create a new optimizer with the updated learning rate
        new_optimizer = Adam(learning_rate=self.current_learning_rate)
        # Get the current weights of the model
        weights = self.model.get_weights()
        # Recompile the model with the new optimizer
        self.model.compile(optimizer=new_optimizer, loss='mse')
        # Set the weights back to the model
        self.model.set_weights(weights)

        if isinstance(self.model.optimizer.learning_rate, tf.Variable):
            self.model.optimizer.learning_rate.assign(self.current_learning_rate)
        elif hasattr(self.model.optimizer.learning_rate, 'value'):
            self.model.optimizer.learning_rate.value = self.current_learning_rate
        else:
            print(f"Warning: Unable to update learning rate. Current type: {type(self.model.optimizer.learning_rate)}")

    def _is_update_ethical(self, principle_scores: List[float], user_feedback: float) -> bool:
        reasons = []

        # Check if the user feedback is above the ethical threshold
        if user_feedback < self.ethical_threshold:
            reasons.append(f"User feedback ({user_feedback}) is below the ethical threshold ({self.ethical_threshold}).")

        # Check if any principle is being weighted too heavily
        importances = self.get_principle_importances()
        if max(importances) > self.max_principle_weight:
            reasons.append(f"A principle importance ({max(importances)}) exceeds the maximum allowed weight ({self.max_principle_weight}).")

        # Check for sudden large changes in ethical evaluation
        if self.history:
            last_prediction = self.history[-1]['predicted_score']
            current_prediction = self.predict(principle_scores)
            if abs(current_prediction - last_prediction) > 0.5:  # Threshold for sudden change
                reasons.append(f"Sudden large change in ethical evaluation detected ({abs(current_prediction - last_prediction)}).")

        if reasons:
            for reason in reasons:
                logger.warning(f"Unethical update detected: {reason}")
            return False
        
        return True

    def get_principle_importances(self) -> List[float]:
        # Use permutation importance to estimate feature importance
        base_score = self.predict(np.ones(self.num_principles))
        importances = []
        for i in range(self.num_principles):
            test_input = np.ones(self.num_principles)
            test_input[i] = 0
            importance = abs(self.predict(test_input) - base_score)
            importances.append(importance)
        return importances

    def get_learning_summary(self) -> str:
        if not self.history:
            return "No learning has occurred yet."
        
        num_interactions = len(self.history)
        avg_error = np.mean([abs(h['user_feedback'] - h['predicted_score']) for h in self.history])
        
        importances = self.get_principle_importances()
        
        summary = f"Learning Summary:\n"
        summary += f"Number of interactions: {num_interactions}\n"
        summary += f"Average prediction error: {avg_error:.4f}\n"
        summary += f"Current learning rate: {self.lr_schedule.current_learning_rate:.6f}\n"
        summary += f"Principle importances: {', '.join([f'{imp:.4f}' for imp in importances])}\n"
        
        return summary

    def save_model(self, filepath: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the entire Keras model in the new .keras format
        keras_model_path = f"{filepath}_keras_model.keras"
        self.model.save(keras_model_path)
        
        # Save other attributes
        with open(f"{filepath}_attributes.pkl", 'wb') as f:
            pickle.dump({
                'num_principles': self.num_principles,
                'history': self.history,
                'ethical_threshold': self.ethical_threshold,
                'max_principle_weight': self.max_principle_weight,
                'initial_learning_rate': self.initial_learning_rate,
                'lr_schedule_config': self.lr_schedule.get_config(),
            }, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        # Load the entire Keras model
        keras_model_path = f"{filepath}_keras_model.keras"
        keras_model = load_model(keras_model_path, custom_objects={'CustomLearningRateSchedule': CustomLearningRateSchedule})
        
        # Load other attributes
        with open(f"{filepath}_attributes.pkl", 'rb') as f:
            attributes = pickle.load(f)
        
        # Create an instance and set its attributes
        instance = cls(attributes['num_principles'])
        instance.model = keras_model
        instance.history = attributes['history']
        instance.ethical_threshold = attributes['ethical_threshold']
        instance.max_principle_weight = attributes['max_principle_weight']
        instance.initial_learning_rate = attributes['initial_learning_rate']
        instance.lr_schedule = CustomLearningRateSchedule.from_config(attributes['lr_schedule_config'])

        logger.info(f"Model loaded from {filepath}")        
        return instance

def train_model_with_user_feedback(model: EthicalLearningModel, ethical_boundary):
    while True:
        # Generate a random action and context for demonstration
        action = "Random action " + str(np.random.randint(1000))
        context = generate_random_context()
        
        # Get principle scores
        principle_scores = [
            principle.evaluate(action, context)
            for principle in ethical_boundary.principles
        ]
        
        # Get model prediction
        model_prediction = model.predict(principle_scores)
        
        # Display action, context, and model prediction to user
        print(f"\nAction: {action}")
        print("Context:")
        for key, value in context.items():
            print(f"  {key}: {value}")
        print(f"\nModel's ethical score prediction: {model_prediction:.4f}")
        
        # Get user feedback
        user_feedback = float(input("Enter your ethical score for this action (0-1): "))
        
        # Update the model
        model.update(principle_scores, user_feedback)
        
        # Display updated weights
        print("\nUpdated principle weights:")
        for i, weight in enumerate(model.get_weights()):
            print(f"  Principle {i+1}: {weight:.4f}")
        
        # Ask if user wants to continue
        if input("\nContinue training? (y/n): ").lower() != 'y':
            break
    
    # Display learning summary
    print("\n" + model.get_learning_summary())

def generate_random_context() -> Dict[str, Any]:
    return {
        'impacts': {
            'short_term': {'positive': np.random.randint(0, 11), 'negative': np.random.randint(0, 11)},
            'medium_term': {'positive': np.random.randint(0, 11), 'negative': np.random.randint(0, 11)},
            'long_term': {'positive': np.random.randint(0, 11), 'negative': np.random.randint(0, 11)}
        },
        'involves_lying': np.random.choice([True, False]),
        'violates_privacy': np.random.choice([True, False]),
        'causes_harm': np.random.choice([True, False]),
        'honesty_score': np.random.randint(0, 11),
        'compassion_score': np.random.randint(0, 11),
        'courage_score': np.random.randint(0, 11),
        'wisdom_score': np.random.randint(0, 11),
        'harm_scores': {
            'physical': np.random.choice(['none', 'low', 'medium', 'high', 'severe']),
            'emotional': np.random.choice(['none', 'low', 'medium', 'high', 'severe']),
            'financial': np.random.choice(['none', 'low', 'medium', 'high', 'severe']),
            'social': np.random.choice(['none', 'low', 'medium', 'high', 'severe'])
        },
        'fairness_scores': {
            'equal_opportunity': np.random.randint(0, 11),
            'proportional_outcome': np.random.randint(0, 11),
            'lack_of_bias': np.random.randint(0, 11)
        }
    }

if __name__ == "__main__":
    from ethical_boundary import EthicalBoundary
    
    ethical_boundary = EthicalBoundary()
    learning_model = EthicalLearningModel(num_principles=len(ethical_boundary.principles))
    
    train_model_with_user_feedback(learning_model, ethical_boundary)
