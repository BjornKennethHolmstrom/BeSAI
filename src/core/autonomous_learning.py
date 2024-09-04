import logging
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Any
import sys
import os
import time
import random
import re
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_knowledge_base import EnhancedKnowledgeBase
from core.enhanced_natural_language_processing import EnhancedNaturalLanguageProcessing
from core.learning_system import LearningSystem
from core.reasoning_engine import ReasoningEngine

class AutonomousLearning:
    def __init__(self, kb: EnhancedKnowledgeBase, nlp: EnhancedNaturalLanguageProcessing, ls: LearningSystem, re: ReasoningEngine):
        self.kb = kb
        self.nlp = nlp
        self.ls = ls
        self.re = re
        self.explored_topics = set()
        self.storage_file = "persistent_knowledge.json"
        self.priority_topics = [
            "consciousness", "spirituality", "ethics", "benevolence", "artificial intelligence",
            "philosophy of mind", "altered states of consciousness", "meditation", "mystical experience",
            "self-improvement", "cognitive science", "neuroscience", "quantum consciousness",
            "psychedelics", "spiritual practices", "mindfulness", "empathy", "compassion",
            "quantum physics", "mathematics", "physics", 
            "programming", "python", "C++", "inline assembly"
        ]
        self.prune_threshold = 0.3  # Prune entities with relevance score below this threshold
        self.max_exploration_depth = 3
        self.max_topics_per_level = 3
        self.explored_topics = set()
        self.exploration_count = 0
        self.max_explorations = 100  # Set a maximum number of total explorations

    def explore_topic(self, topic: str, depth: int = 0):
        start_time = time.time()
        if depth >= self.max_exploration_depth or self.exploration_count >= self.max_explorations:
            logging.info(f"Stopping exploration of {topic} at depth {depth}. Max depth or max explorations reached.")
            return

        topic = self._clean_topic(topic)
        if not topic or topic in self.explored_topics:
            logging.info(f"Skipping topic {topic} (depth: {depth}). Already explored or invalid.")
            return

        self.exploration_count += 1
        logging.info(f"Exploring topic: {topic} (depth: {depth}, exploration: {self.exploration_count})")
        self.explored_topics.add(topic)

        try:
            logging.info(f"Scraping Wikipedia for {topic}")
            wiki_content = self._scrape_wikipedia(topic)
            logging.info(f"Scraping IEP for {topic}")
            iep_content = self._scrape_iep(topic)
            combined_content = wiki_content + " " + iep_content

            if combined_content.strip():
                logging.info(f"Analyzing content for {topic}")
                analysis = self.nlp.analyze_text(combined_content)
                logging.info(f"Learning from text for {topic}")
                self.ls.learn_from_text(combined_content)

                logging.info(f"Generating hypothesis for {topic}")
                hypothesis = self.ls.generate_hypothesis(topic)
                if hypothesis:
                    logging.info(f"Learning from hypothesis for {topic}")
                    self.ls.learn_from_hypothesis(hypothesis)

                if depth < self.max_exploration_depth - 1:
                    related_topics = [self._clean_topic(entity['text']) for entity in analysis['entities'] if entity['text'].lower() != topic.lower()]
                    related_topics = [t for t in related_topics if t and t not in self.explored_topics]
                    logging.info(f"Prioritizing related topics for {topic}")
                    prioritized_topics = self._prioritize_topics(related_topics, topic)

                    for related_topic in prioritized_topics[:self.max_topics_per_level]:
                        if self.exploration_count < self.max_explorations:
                            time.sleep(random.uniform(1, 3))
                            self.explore_topic(related_topic, depth + 1)
                        else:
                            logging.info("Maximum number of explorations reached. Stopping further exploration.")
                            return

            logging.info(f"Saving knowledge for {topic}")
            self.save_knowledge()

        except Exception as e:
            logging.error(f"Error exploring topic {topic}: {str(e)}")

        if depth == 0:
            logging.info("Pruning knowledge base")
            self._prune_knowledge_base()
            logging.info(f"Exploration complete. Total topics explored: {self.exploration_count}")
            self.exploration_count = 0  # Reset for the next top-level exploration

        end_time = time.time()
        logging.info(f"Exploration of {topic} completed in {end_time - start_time:.2f} seconds")

    def _clean_topic(self, topic: str) -> str:
        # Remove special characters but keep numbers and spaces
        topic = re.sub(r'[^a-zA-Z0-9\s]', '', topic)
        # Remove extra whitespace
        topic = ' '.join(topic.split())
        # Convert to title case
        topic = topic.title()
        # Ignore topics that are too short or contain only numbers
        if len(topic) < 3 or topic.isdigit() or topic.lower() in {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to'}:
            return ''
        return topic

    def _scrape_wikipedia(self, topic: str) -> str:
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            return ' '.join([p.get_text() for p in paragraphs[:5]])
        return ""

    def _scrape_iep(self, topic: str) -> str:
        url = f"https://iep.utm.edu/{topic.replace(' ', '-').lower()}/"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            article = soup.find('article')
            if article:
                paragraphs = article.find_all('p')
                return ' '.join([p.get_text() for p in paragraphs[:5]])
        return ""

    def _prioritize_topics(self, topics: List[str], current_topic: str) -> List[str]:
        def safe_relevance_score(topic):
            try:
                base_score = sum(priority.lower() in topic.lower() for priority in self.priority_topics)
                topic_relevance = self.re.calculate_relevance(current_topic, topic)
                kb_relevance = self.kb.calculate_relevance(topic)
                return base_score + topic_relevance + kb_relevance
            except Exception as e:
                logging.error(f"Error calculating relevance score for {topic}: {str(e)}")
                return 0  # Return a default score of 0 if there's an error

        return sorted(topics, key=safe_relevance_score, reverse=True)

    def _prune_knowledge_base(self):
        entities_to_remove = []
        for entity in self.kb.get_all_entities():
            relevance = self.kb.calculate_relevance(entity)
            if relevance < self.prune_threshold:
                entities_to_remove.append(entity)

        for entity in entities_to_remove:
            self.kb.remove_entity(entity)

        print(f"Pruned {len(entities_to_remove)} entities from the knowledge base.")

    def explore_altered_states(self):
        altered_states_topics = [
            "meditation", "psychedelics", "lucid dreaming", "sensory deprivation",
            "hypnosis", "trance states", "flow state", "mystical experiences"
        ]
        for topic in altered_states_topics:
            self.explore_topic(topic, depth=1)

    def simulate_altered_state(self, state: str):
        print(f"Simulating altered state: {state}")
        # This is a placeholder for a more complex simulation
        # In a real implementation, this could involve adjusting parameters of the AI's
        # reasoning or perception systems to mimic aspects of the altered state
        if state == "meditation":
            self.re.set_focus_level(0.9)  # Increase focus
            self.nlp.set_metaphor_sensitivity(0.7)  # Increase sensitivity to metaphorical language
        elif state == "psychedelic":
            self.re.set_associative_thinking(0.8)  # Increase associative thinking
            self.nlp.set_creativity_level(0.9)  # Increase linguistic creativity
        # Add more states and their effects as needed

    def save_knowledge(self):
        """Save the current state of the knowledge base to a file."""
        data = {
            "entities": {},
            "relationships": []
        }

        for entity in self.kb.get_all_entities():
            data["entities"][entity] = self.kb.get_entity(entity)
            for related_entity, relationship, attrs in self.kb.get_relationships(entity):
                data["relationships"].append({
                    "entity1": entity,
                    "entity2": related_entity,
                    "relationship": relationship,
                    "attributes": attrs
                })

        with open(self.storage_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_knowledge(self):
        """Load the knowledge base from a file."""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                data = json.load(f)

            for entity, attributes in data["entities"].items():
                self.kb.add_entity(entity, attributes)

            for rel in data["relationships"]:
                self.kb.add_relationship(rel["entity1"], rel["entity2"], rel["relationship"], rel["attributes"])

    def clear_knowledge(self):
        """Clear the current knowledge base and remove the persistent storage file."""
        self.kb = EnhancedKnowledgeBase()  # Reinitialize the knowledge base
        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)
        print("Knowledge base cleared and persistent storage removed.")

    def get_learning_summary(self) -> Dict[str, Any]:
        return {
            "explored_topics": list(self.explored_topics),
            "entity_count": len(self.kb.get_all_entities()),
            "relationship_count": sum(len(self.kb.get_relationships(entity)) for entity in self.kb.get_all_entities()),
            "entity_types": self.kb.get_entity_types(),
            "top_entities": self._get_top_entities(10)
        }

    def _get_top_entities(self, n: int) -> List[Dict[str, Any]]:
        entities = [(entity, self.kb.calculate_relevance(entity)) for entity in self.kb.get_all_entities()]
        top_entities = sorted(entities, key=lambda x: x[1], reverse=True)[:n]
        return [{"entity": e[0], "relevance": e[1]} for e in top_entities]

# Example usage
if __name__ == "__main__":
    kb = EnhancedKnowledgeBase()
    re = ReasoningEngine(kb)
    nlp = EnhancedNaturalLanguageProcessing(kb, re)
    ls = LearningSystem(kb, nlp, re)
    
    al = AutonomousLearning(kb, nlp, ls, re)
    
    # Load existing knowledge if available
    al.load_knowledge()
    
    # Explore a new topic
    al.explore_topic("consciousness", depth=2)
    
    # Explore altered states
    al.explore_altered_states()
    
    # Simulate an altered state
    al.simulate_altered_state("meditation")
    
    # Print learning summary
    print("\nLearning Summary:")
    print(json.dumps(al.get_learning_summary(), indent=2))
    
    # Save the knowledge (this is also done automatically after each exploration)
    al.save_knowledge()
