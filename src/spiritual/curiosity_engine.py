import random
from typing import List, Dict, Any
import logging

class CuriosityEngine:
    def __init__(self, autonomous_learning):
        self.autonomous_learning = autonomous_learning
        self.interest_areas = {}
        self.exploration_history = []

    def generate_curiosity_prompt(self) -> Dict[str, str]:
        strategy = self.get_exploration_strategy()
        if strategy == "explore_new" or not self.interest_areas:
            topic = self.autonomous_learning.select_next_topic()
        else:
            topic = max(self.interest_areas, key=self.interest_areas.get)

        question = self.generate_question(topic)
        self.exploration_history.append((topic, question))

        return {
            "topic": topic,
            "question": question,
            "strategy": strategy
        }

    def generate_question(self, topic: str) -> str:
        templates = [
            f"What is the relationship between {topic} and {{}}?",
            f"How does {topic} affect {{}}?",
            f"What are the key components of {topic}?",
            f"Can you explain the process of {topic}?",
            f"What are the applications of {topic} in {{}}?"
        ]
        try:
            related_topics = self.autonomous_learning.kb.get_related_topics(topic)
            if related_topics:
                return random.choice(templates).format(random.choice(related_topics))
        except Exception as e:
            logging.warning(f"Error getting related topics for {topic}: {str(e)}")
        
        return random.choice(templates).format("other fields")

    def get_exploration_strategy(self) -> str:
        if random.random() < 0.3:  # 30% chance to explore a new topic
            return "explore_new"
        else:
            return "deepen_existing"

    def update_interest_model(self, topic: str, interest_score: float):
        if topic in self.interest_areas:
            self.interest_areas[topic] = (self.interest_areas[topic] + interest_score) / 2
        else:
            self.interest_areas[topic] = interest_score

    def suggest_exploration(self) -> str:
        try:
            prompt = self.generate_curiosity_prompt()
            return prompt['topic']
        except Exception as e:
            logging.error(f"Error suggesting exploration: {str(e)}")
            return self.get_fallback_topic()

    def get_fallback_topic(self) -> str:
        fallback_topics = [
            "artificial intelligence",
            "consciousness",
            "ethics",
            "philosophy",
            "technology"
        ]
        return random.choice(fallback_topics)

    def process_exploration_result(self, topic: str, exploration_result: Dict[str, Any]):
        try:
            # Update interest model based on exploration result
            interest_score = len(exploration_result.get('entities', [])) + len(exploration_result.get('relationships', []))
            self.update_interest_model(topic, interest_score / 10)  # Normalize score
        except Exception as e:
            logging.error(f"Error processing exploration result: {str(e)}")

    def get_curiosity_metrics(self) -> Dict[str, float]:
        try:
            return {
                "total_topics_explored": len(self.exploration_history),
                "average_interest_score": sum(self.interest_areas.values()) / len(self.interest_areas) if self.interest_areas else 0,
                "exploration_diversity": len(set(topic for topic, _ in self.exploration_history)) / len(self.exploration_history) if self.exploration_history else 0
            }
        except Exception as e:
            logging.error(f"Error calculating curiosity metrics: {str(e)}")
            return {}
