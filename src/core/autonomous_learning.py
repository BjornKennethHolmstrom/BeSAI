import logging
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Any
import sys
import os
import time
import random
import threading
import schedule
import re
from collections import Counter
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_knowledge_base import EnhancedKnowledgeBase
from core.enhanced_natural_language_processing import EnhancedNaturalLanguageProcessing
from core.learning_system import LearningSystem
from core.reasoning_engine import ReasoningEngine
from core.meta_cognition import Metacognition

class AutonomousLearning:
    def __init__(self, kb: EnhancedKnowledgeBase, nlp: EnhancedNaturalLanguageProcessing, ls: LearningSystem, re: ReasoningEngine, metacognition: Metacognition):
        self.kb = kb
        self.nlp = nlp
        self.ls = ls
        self.re = re
        self.metacognition = metacognition
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
        self.exploration_thread = None
        self.exploration_interval = 60  # Default to 1 minutes
        self.save_schedule_thread = None
        self.stop_exploration = threading.Event()
        self.stop_event = threading.Event()
        self.check_interval = 1 # Check for stop every 1 second
        self.learning_plan_interval = timedelta(days=7)  # Generate a new plan weekly
        self.last_plan_generation = datetime.min

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

                # Add knowledge to the versioned knowledge base
                self.kb.add_entity(topic, {"content": combined_content}, source="Wikipedia and IEP")
                
                for entity in analysis['entities']:
                    self.kb.add_entity(entity['text'], {"type": entity['label']}, source="NLP Analysis")
                
                for relationship in analysis['relationships']:
                    self.kb.add_relationship(relationship['subject'], relationship['object'], relationship['predicate'], source="NLP Analysis")

                # Perform metacognitive assessment
                self.perform_metacognitive_assessment(topic)

                logging.info(f"Generating hypothesis for {topic}")
                hypothesis = self.ls.generate_hypothesis(topic)
                if hypothesis:
                    logging.info(f"Generated hypothesis for {topic}: {hypothesis[:100]}...")  # Log first 100 chars
                    self.kb.add_entity(f"hypothesis_{topic}", {"content": hypothesis}, source="Hypothesis Generation")
                    logging.info(f"Stored hypothesis for {topic}")

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
            self.metacognition.update_goal_progress(topic)
            logging.info(f"Saving knowledge for {topic}")
            self.save_knowledge()

        except Exception as e:
            logging.error(f"Error exploring topic {topic}: {str(e)}", exc_info=True)
            self.kb.flag_improvement(topic, f"Error during exploration: {str(e)}")

        end_time = time.time()
        logging.info(f"Exploration of {topic} completed in {end_time - start_time:.2f} seconds")

    def autonomous_exploration(self):
        while not self.stop_event.is_set():
            self.check_and_generate_learning_plan()
            self.pursue_learning_goals()
            
            # Existing exploration logic
            if len(self.explored_topics) < self.max_explorations:
                topic = self.select_next_topic()
                if topic:
                    self.explore_topic(topic)
                else:
                    logging.info("No unexplored topics found. Waiting...")
            self.wait(self.exploration_interval)

    def check_and_generate_learning_plan(self):
        if datetime.now() - self.last_plan_generation >= self.learning_plan_interval:
            plan = self.metacognition.generate_learning_plan()
            logging.info("Generated new learning plan:\n" + plan)
            self.last_plan_generation = datetime.now()

    def pursue_learning_goals(self):
        active_goals = self.metacognition.get_active_goals()
        for goal in active_goals:
            topic = goal['topic']
            if topic not in self.explored_topics:
                self.explore_topic(topic)
            self.metacognition.update_goal_progress(topic)

    def select_next_topic(self):
        # Prioritize topics from learning goals
        active_goals = self.metacognition.get_active_goals()
        goal_topics = [goal['topic'] for goal in active_goals if goal['topic'] not in self.explored_topics]
        if goal_topics:
            return random.choice(goal_topics)

        # Fall back to existing topic selection logic
        unexplored_priorities = [topic for topic in self.priority_topics if topic not in self.explored_topics]
        if unexplored_priorities:
            return random.choice(unexplored_priorities)
        
        all_entities = set(self.kb.get_all_entities())
        unexplored_topics = all_entities - self.explored_topics
        return random.choice(list(unexplored_topics)) if unexplored_topics else None

    def wait(self, duration):
        """Wait for the specified duration, checking for stop event periodically"""
        end_time = time.time() + duration
        while time.time() < end_time and not self.stop_event.is_set():
            time.sleep(min(self.check_interval, end_time - time.time()))

    def start_autonomous_exploration(self):
        if not self.is_exploring():
            self.stop_event.clear()
            self.exploration_thread = threading.Thread(target=self.autonomous_exploration)
            self.exploration_thread.start()
            self.start_periodic_saving()
            logging.info("Autonomous exploration started.")
        else:
            logging.info("Autonomous exploration is already running.")

    def _clean_topic(self, topic: str) -> str:
        # Remove special characters but keep spaces
        topic = re.sub(r'[^a-zA-Z0-9\s]', '', topic)
        # Remove extra whitespace
        topic = ' '.join(topic.split())
        # Convert to title case
        topic = topic.title()
        # Remove any numbers at the end of the topic
        topic = re.sub(r'\d+$', '', topic).strip()
        # Ignore topics that are too short or only contain numbers
        if len(topic) < 3 or topic.isdigit() or topic.lower() in {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to'}:
            return ''
        return topic

    def stop_autonomous_exploration(self):
        logging.info("Stopping autonomous exploration...")
        self.stop_event.set()
        
        if self.exploration_thread and self.exploration_thread.is_alive():
            self.exploration_thread.join(timeout=10)
        
        if self.save_schedule_thread and self.save_schedule_thread.is_alive():
            self.save_schedule_thread.join(timeout=10)
        
        self.exploration_thread = None
        self.save_schedule_thread = None
        logging.info("Autonomous exploration stopped.")

    def is_exploring(self):
        return self.exploration_thread is not None and self.exploration_thread.is_alive()

    def set_exploration_interval(self, interval: int):
        if interval < 1:
            raise ValueError("Exploration interval must be at least 1 second.")
        self.exploration_interval = interval
        logging.info(f"Exploration interval set to {interval} seconds.")

    def add_priority_topic(self, topic: str):
        if topic not in self.priority_topics:
            self.priority_topics.append(topic)
            logging.info(f"Added {topic} to priority topics")

    def remove_priority_topic(self, topic: str):
        if topic in self.priority_topics:
            self.priority_topics.remove(topic)
            logging.info(f"Removed {topic} from priority topics")

    def get_priority_topics(self):
        return self.priority_topics

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

    def prioritize_topic(self, topic: str):
        if topic not in self.priority_topics:
            self.priority_topics.append(topic)
            logging.info(f"Added {topic} to priority topics due to low confidence")

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

    def address_knowledge_gaps(self, topic: str, gaps: List[str]):
        for gap in gaps:
            subtopic = f"{topic}_{gap.replace('Missing ', '')}"
            if subtopic not in self.explored_topics:
                self.priority_topics.append(subtopic)
                logging.info(f"Added {subtopic} to priority topics to address knowledge gap")

    def _prune_knowledge_base(self):
        entities_to_remove = []
        for entity in self.kb.get_all_entities():
            relevance = self.kb.calculate_relevance(entity)
            if relevance < self.prune_threshold:
                entities_to_remove.append(entity)

        for entity in entities_to_remove:
            self.kb.remove_entity(entity)

        print(f"Pruned {len(entities_to_remove)} entities from the knowledge base.")

    def perform_metacognitive_assessment(self, topic: str):
        logging.info(f"Performing metacognitive assessment for {topic}")
        assessment = self.metacognition.assess_knowledge(topic)
        learning_analysis = self.metacognition.analyze_learning_process(topic)

        # Use metacognitive insights to guide further learning
        if assessment['confidence'] < self.metacognition.confidence_threshold:
            logging.info(f"Confidence in {topic} is low. Prioritizing further exploration.")
            self.prioritize_topic(topic)

        if assessment['gaps']:
            logging.info(f"Identified knowledge gaps for {topic}: {', '.join(assessment['gaps'])}")
            self.address_knowledge_gaps(topic, assessment['gaps'])

        # Generate and store meta-insight
        meta_insight = self.metacognition.generate_meta_insight(topic)
        self.kb.add_entity(f"meta_insight_{topic}", {"content": meta_insight}, source="Metacognition")

    def get_insight(self, topic: str) -> str:
        entity = self.kb.get_entity(topic)
        if not entity:
            return f"No information available for the topic: {topic}. This topic has not been explored yet."

        insight = f"Insight on {topic}:\n"
        
        # Add attributes
        attributes = [f"  {key}: {value}" for key, value in entity.items() if key != 'metadata']
        if attributes:
            insight += "Attributes:\n" + "\n".join(attributes) + "\n"
        else:
            insight += "No specific attributes found for this topic.\n"

        # Add relationships
        relationships = self.kb.get_relationships(topic)
        if relationships:
            insight += "Relationships:\n"
            for related_entity, relationship_type, attrs in relationships:
                insight += f"  {relationship_type} -> {related_entity}\n"
        else:
            insight += "No relationships found for this topic.\n"

        # Add relevance score
        relevance = self.kb.calculate_relevance(topic)
        insight += f"Relevance score: {relevance:.2f}\n"

        # Add hypothesis to insight if available
        hypothesis_entity = self.kb.get_entity(f"hypothesis_{topic}")
        if hypothesis_entity:
            hypothesis = hypothesis_entity.get("content", "No hypothesis available.")
            insight += f"\n\nHypothesis:\n{hypothesis}"
     
        if not attributes and not relationships and relevance <= 0:
            insight += "\nNote: This topic exists in our knowledge base, but we don't have much meaningful information about it yet. Further exploration may be needed."

        # Add meta-insight
        meta_insight_entity = self.kb.get_entity(f"meta_insight_{topic}")
        if meta_insight_entity:
            meta_insight = meta_insight_entity.get("content", "No meta-insight available.")
            insight += f"\n\nMeta Insight:\n{meta_insight}"
        else:
            insight += "\n\nNo meta-insight available for this topic yet."

        return insight

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

    def save_knowledge_periodically(self):
        self.kb.increment_version()
        self.kb.save_to_file(f"knowledge_base_v{self.kb.version}.json")
        logging.info(f"Knowledge base saved as version {self.kb.version}.")

    def start_periodic_saving(self):
        if not self.save_schedule_thread or not self.save_schedule_thread.is_alive():
            self.save_schedule_thread = threading.Thread(target=self.periodic_saving)
            self.save_schedule_thread.start()

    def periodic_saving(self):
        while not self.stop_event.is_set():
            self.save_knowledge()
            self.wait(1800)  # 30 minutes

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

    def set_exploration_interval(self, seconds: int):
        self.exploration_interval = seconds

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
