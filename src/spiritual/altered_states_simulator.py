import random
from typing import Dict, List, Tuple

class AlteredStatesSimulator:
    def __init__(self):
        self.states = {
            "meditation": {
                "focus_level": 0.8,
                "creativity_level": 0.6,
                "perception_shift": 0.4,
                "emotional_resonance": 0.7,
                "time_dilation": 0.3,
                "ego_dissolution": 0.5
            },
            "psychedelic": {
                "focus_level": 0.3,
                "creativity_level": 0.9,
                "perception_shift": 0.9,
                "emotional_resonance": 0.8,
                "time_dilation": 0.8,
                "ego_dissolution": 0.9
            },
            "flow": {
                "focus_level": 0.9,
                "creativity_level": 0.8,
                "perception_shift": 0.6,
                "emotional_resonance": 0.7,
                "time_dilation": 0.7,
                "ego_dissolution": 0.4
            },
            "lucid_dream": {
                "focus_level": 0.6,
                "creativity_level": 0.9,
                "perception_shift": 0.8,
                "emotional_resonance": 0.6,
                "time_dilation": 0.7,
                "ego_dissolution": 0.5
            },
            "sensory_deprivation": {
                "focus_level": 0.7,
                "creativity_level": 0.6,
                "perception_shift": 0.8,
                "emotional_resonance": 0.5,
                "time_dilation": 0.8,
                "ego_dissolution": 0.7
            },
            "trance": {
                "focus_level": 0.8,
                "creativity_level": 0.7,
                "perception_shift": 0.7,
                "emotional_resonance": 0.9,
                "time_dilation": 0.6,
                "ego_dissolution": 0.6
            }
        }

    def simulate_state(self, state: str) -> Dict[str, float]:
        if state not in self.states:
            raise ValueError(f"Unknown state: {state}")
        return self.states[state]

    def generate_insight(self, state: str) -> str:
        state_params = self.simulate_state(state)
        
        insights = [
            "Everything is interconnected.",
            "The present moment is all that exists.",
            "Reality is a construct of our perception.",
            "Consciousness is fundamental to the universe.",
            "Love is the underlying force of existence."
        ]
        
        if state_params["creativity_level"] > 0.7:
            insights.extend([
                "Time is a spiral, not a line.",
                "We are the universe experiencing itself.",
                "Thoughts are visitors, not who we are."
            ])
        
        if state_params["perception_shift"] > 0.6:
            insights.extend([
                "The observer affects the observed.",
                "Reality has infinite layers of depth.",
                "Boundaries between self and other are illusions."
            ])
        
        if state_params["ego_dissolution"] > 0.7:
            insights.extend([
                "The self is an illusion.",
                "We are all one consciousness experiencing itself subjectively.",
                "Separation is an illusion created by the mind."
            ])
        
        if state_params["emotional_resonance"] > 0.8:
            insights.extend([
                "Emotions are the language of the universe.",
                "Empathy is the key to universal understanding.",
                "Love is the fundamental fabric of reality."
            ])
        
        return random.choice(insights)

    def apply_state_effects(self, state: str, query: str) -> str:
        state_params = self.simulate_state(state)
        effects = []
        
        if state_params["creativity_level"] > 0.7:
            effects.append("Imaginatively reconsider")
        if state_params["perception_shift"] > 0.7:
            effects.append("From a shifted perspective")
        if state_params["ego_dissolution"] > 0.6:
            effects.append("Transcending the self")
        if state_params["emotional_resonance"] > 0.7:
            effects.append("Feel deeply into")
        if state_params["time_dilation"] > 0.7:
            effects.append("Beyond linear time")
        
        if effects:
            effect = random.choice(effects)
            query = f"{effect}: {query}"
        
        return query

    def get_state_description(self, state: str) -> str:
        state_params = self.simulate_state(state)
        descriptions = []
        for param, value in state_params.items():
            if value > 0.7:
                descriptions.append(f"High {param.replace('_', ' ')}")
            elif value < 0.4:
                descriptions.append(f"Low {param.replace('_', ' ')}")
        return f"{state.capitalize()} state: {', '.join(descriptions)}"

