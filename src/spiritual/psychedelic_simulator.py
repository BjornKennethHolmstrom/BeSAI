# BeSAI/src/spiritual/psychedelics_simulator.py

import random
from typing import Dict, List, Tuple

class PsychedelicSimulator:
    def __init__(self):
        self.intensity_levels = {
            "threshold": 0.2,
            "light": 0.4,
            "medium": 0.6,
            "strong": 0.8,
            "intense": 1.0
        }
        self.current_intensity = "medium"
        self.duration = 0
        self.peak_duration = 0
        self.substances = {
            "psilocybin": {
                "visual_effects": 0.7,
                "cognitive_effects": 0.8,
                "emotional_effects": 0.9,
                "duration": 6,
                "peak_duration": 2
            },
            "lsd": {
                "visual_effects": 0.9,
                "cognitive_effects": 0.9,
                "emotional_effects": 0.7,
                "duration": 12,
                "peak_duration": 3
            },
            "dmt": {
                "visual_effects": 1.0,
                "cognitive_effects": 1.0,
                "emotional_effects": 1.0,
                "duration": 0.25,
                "peak_duration": 0.1
            },
            "mescaline": {
                "visual_effects": 0.6,
                "cognitive_effects": 0.7,
                "emotional_effects": 0.8,
                "duration": 10,
                "peak_duration": 4
            }
        }
        self.current_substance = None

    def set_substance(self, substance: str):
        if substance in self.substances:
            self.current_substance = substance
            self.duration = self.substances[substance]["duration"]
            self.peak_duration = self.substances[substance]["peak_duration"]
        else:
            raise ValueError(f"Unknown substance: {substance}")

    def set_intensity(self, intensity: str):
        if intensity in self.intensity_levels:
            self.current_intensity = intensity
        else:
            raise ValueError(f"Unknown intensity level: {intensity}")

    def simulate_effects(self) -> Dict[str, float]:
        if not self.current_substance:
            raise ValueError("No substance selected. Use set_substance() first.")

        base_effects = self.substances[self.current_substance]
        intensity_factor = self.intensity_levels[self.current_intensity]

        return {
            "visual_distortion": base_effects["visual_effects"] * intensity_factor,
            "cognitive_flexibility": base_effects["cognitive_effects"] * intensity_factor,
            "emotional_amplification": base_effects["emotional_effects"] * intensity_factor,
            "ego_dissolution": random.uniform(0.5, 1.0) * intensity_factor,
            "time_perception_alteration": random.uniform(0.6, 1.0) * intensity_factor,
            "synesthesia": random.uniform(0.3, 0.8) * intensity_factor
        }

    def generate_visual_description(self) -> str:
        effects = self.simulate_effects()
        visual_intensity = effects["visual_distortion"]
        
        descriptions = [
            "Colors seem more vibrant and alive",
            "Patterns and textures breathe and flow",
            "Objects leave trailing afterimages",
            "The boundaries between objects blur and merge",
            "Fractal patterns emerge in everyday surfaces",
            "Reality seems to ripple and wave",
            "Closed-eye visuals reveal intricate geometric patterns",
            "The world appears to be made of shimmering energy"
        ]
        
        selected_descriptions = random.sample(descriptions, k=min(3, max(1, int(visual_intensity * len(descriptions)))))
        return " ".join(selected_descriptions) + "."

    def generate_cognitive_insight(self) -> str:
        effects = self.simulate_effects()
        cognitive_intensity = effects["cognitive_flexibility"]
        
        insights = [
            "Everything is interconnected in a vast cosmic web",
            "Time is an illusion, only the present moment exists",
            "Consciousness is the fundamental fabric of reality",
            "The self is a construct, we are all one",
            "Language is limiting, true understanding transcends words",
            "Reality is a projection of collective consciousness",
            "Every moment contains infinite potential",
            "Love is the underlying force that binds the universe"
        ]
        
        return random.choice(insights) if random.random() < cognitive_intensity else "No significant insight at this moment."

    def apply_psychedelic_filter(self, text: str) -> str:
        effects = self.simulate_effects()
        
        # Apply cognitive flexibility
        if random.random() < effects["cognitive_flexibility"]:
            words = text.split()
            random.shuffle(words)
            text = " ".join(words)
        
        # Apply emotional amplification
        if random.random() < effects["emotional_amplification"]:
            text = text.upper() + "!!!"
        
        # Apply time perception alteration
        if random.random() < effects["time_perception_alteration"]:
            text = text.replace(" ", "...") + " (time dilates)"
        
        # Apply synesthesia
        if random.random() < effects["synesthesia"]:
            colors = ["red", "blue", "green", "yellow", "purple"]
            text += f" (tastes like {random.choice(colors)})"
        
        return text

    def generate_experience_report(self) -> str:
        effects = self.simulate_effects()
        
        report = f"Substance: {self.current_substance.capitalize()}\n"
        report += f"Intensity: {self.current_intensity.capitalize()}\n\n"
        report += f"Visual Effects: {self.generate_visual_description()}\n\n"
        report += f"Cognitive Insight: {self.generate_cognitive_insight()}\n\n"
        report += "Perceived Reality: " + self.apply_psychedelic_filter("Reality is a construct of our perception")
        
        return report

# Example usage
if __name__ == "__main__":
    simulator = PsychedelicSimulator()
    simulator.set_substance("psilocybin")
    simulator.set_intensity("medium")
    print(simulator.generate_experience_report())
