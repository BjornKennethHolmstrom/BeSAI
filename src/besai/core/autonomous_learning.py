import logging
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Any, Optional
import sys
import os
import time
import random
import threading
import schedule
import re
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_knowledge_base import EnhancedKnowledgeBase
from core.enhanced_natural_language_processing import EnhancedNaturalLanguageProcessing
from core.learning_system import LearningSystem
from core.reasoning_engine import ReasoningEngine
from core.meta_cognition import Metacognition
from besai.logging_config import setup_logging

setup_logging()

class AutonomousLearning:
    def __init__(self, kb: EnhancedKnowledgeBase, nlp: EnhancedNaturalLanguageProcessing, ls: LearningSystem, re: ReasoningEngine, me: Metacognition):
        self.logger = logging.getLogger(__name__)
        self.kb = kb
        self.nlp = nlp
        self.ls = ls
        self.re = re
        self.metacognition = me
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
        self.current_topic = None

        self.clean_priority_topics()

    def reset_knowledge_base(self):
        logger.info("Resetting knowledge base and explored topics")
        self.kb = EnhancedKnowledgeBase()
        self.explored_topics = set()
        self.exploration_count = 0
        self.last_plan_generation = datetime.min

    def _retrieve_content(self, topic: str) -> str:
        content = self._scrape_wikipedia(topic)
        if not content:
            content = self._scrape_iep(topic)
        if not content:
            content = self._search_and_scrape(topic)  # New method to search and scrape other sources
        return content

    def _search_and_scrape(self, topic: str) -> str:
        search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract and visit the first few search result links
            links = soup.select('.yuRUbf > a')[:3]  # Adjust the number of links as needed
            content = ""
            
            for link in links:
                url = link['href']
                try:
                    page_response = requests.get(url, headers=headers, timeout=5)
                    page_response.raise_for_status()
                    page_soup = BeautifulSoup(page_response.text, 'html.parser')
                    paragraphs = page_soup.find_all('p')
                    content += ' '.join([p.get_text() for p in paragraphs[:10]])  # Increased from 5 to 10
                except Exception as e:
                    logger.error(f"Error scraping {url}: {str(e)}")
            
            if not content:
                logger.warning(f"No content found for topic: {topic}")
                content = f"No detailed information found for {topic}. This topic may require further research."
            
            return content.strip()
        except Exception as e:
            logger.error(f"Error searching for {topic}: {str(e)}")
            return f"Error occurred while searching for information on {topic}."

    def _smart_group_tokens(self, tokens: List[str]) -> List[str]:
        grouped_tokens = []
        for i in range(1, len(tokens) + 1):
            grouped_tokens.extend([' '.join(combo) for combo in combinations(tokens, i)])
        return sorted(grouped_tokens, key=len, reverse=True)

    def explore_topic(self, topic: str, depth: int = 0):
        if depth == 0:
            self.reset_knowledge_base()

        start_time = time.time()
        logger.info(f"Starting exploration of topic: {topic} (depth: {depth})")

        try:
            if depth >= self.max_exploration_depth or self.exploration_count >= self.max_explorations:
                logger.info(f"Stopping exploration of {topic} at depth {depth}. Max depth or max explorations reached.")
                return

            cleaned_topic = self._clean_topic(topic)
            self.current_topic = cleaned_topic
            if not cleaned_topic:
                logger.info(f"Skipping invalid topic: {topic}")
                return

            if cleaned_topic in self.explored_topics:
                logger.info(f"Topic {cleaned_topic} already explored. Skipping.")
                return

            self.exploration_count += 1
            self.explored_topics.add(cleaned_topic)
            content = self._retrieve_content(cleaned_topic)
            if not content:
                logger.warning(f"No content found for topic: {cleaned_topic}")
                content = f"No detailed information found for {cleaned_topic}. This topic may require further research."

            logger.info(f"Content found for topic: {cleaned_topic}. Length: {len(content)}")

            if content:
                analysis = self.nlp.analyze_text(content)
                self.ls.learn_from_text(content)
                
                # Add knowledge to the knowledge base
                self.kb.add_entity(cleaned_topic, {"content": content}, source="Web Exploration")
                
                for entity in analysis.get('entities', []):
                    self.kb.add_entity(entity['text'], {"type": entity.get('label', 'unknown')}, source="NLP Analysis")
                
                for relationship in analysis.get('relationships', []):
                    self.kb.add_relationship(
                        relationship.get('subject', ''), 
                        relationship.get('object', ''), 
                        relationship.get('predicate', ''), 
                        source="NLP Analysis"
                    )

                # Use metacognition to assess knowledge and generate insights
                self.perform_metacognitive_assessment(cleaned_topic)

                # Generate and store hypothesis
                hypothesis = self.ls.generate_hypothesis(cleaned_topic)
                if hypothesis:
                    self.kb.add_entity(f"hypothesis_{cleaned_topic}", {"content": hypothesis}, source="Hypothesis Generation")

                # Use metacognition to set learning goals
                self.set_learning_goals(cleaned_topic)

            analysis = self.nlp.analyze_text(content)
            self.ls.learn_from_text(content)

            logger.info(f"Analysis complete. Entities found: {len(analysis.get('entities', []))}, Relationships found: {len(analysis.get('relationships', []))}")

            # Add knowledge to the knowledge base
            self.kb.add_entity(cleaned_topic, {"content": content}, source="Web Exploration")
            
            for entity in analysis.get('entities', []):
                self.kb.add_entity(entity['text'], {"type": entity.get('label', 'unknown')}, source="NLP Analysis")
            
            for relationship in analysis.get('relationships', []):
                self.kb.add_relationship(
                    relationship.get('subject', ''), 
                    relationship.get('object', ''), 
                    relationship.get('predicate', ''), 
                    source="NLP Analysis"
                )

            self.perform_metacognitive_assessment(cleaned_topic)

            hypothesis = self.ls.generate_hypothesis(cleaned_topic)
            if hypothesis:
                self.kb.add_entity(f"hypothesis_{cleaned_topic}", {"content": hypothesis}, source="Hypothesis Generation")

            if depth < self.max_exploration_depth - 1:
                related_topics = self._extract_related_topics(analysis)
                prioritized_topics = self._prioritize_topics(related_topics, cleaned_topic)

                for related_topic in prioritized_topics[:self.max_topics_per_level]:
                    if self.exploration_count < self.max_explorations and not self.stop_event.is_set():
                        time.sleep(random.uniform(1, 3))
                        self.explore_topic(related_topic, depth + 1)
                    else:
                        logger.info("Maximum explorations reached or stop event set. Stopping further exploration.")
                        break

            return f"Explored topic: {cleaned_topic}. Added knowledge to the knowledge base."

        except Exception as e:
            logger.error(f"Error exploring topic {topic}: {str(e)}", exc_info=True)
            return f"Error during exploration: {str(e)}"

        finally:
            end_time = time.time()
            logger.info(f"Exploration of {topic} completed in {end_time - start_time:.2f} seconds")

    def set_learning_goals(self, topic: str):
        plan = self.metacognition.generate_learning_plan()
        logging.info(f"Generated learning plan for {topic}: {plan}")

        # Implement the learning plan
        for goal in self.metacognition.get_active_goals():
            if goal['topic'] == topic:
                self.prioritize_topic(topic)
                # You can add more specific actions based on the goal type

    def _explore_subtopic(self, subtopic: str, depth: int) -> Optional[str]:
        logger.info(f"Exploring subtopic: {subtopic} (depth: {depth})")

        content = self._retrieve_content(subtopic)
        if not content:
            logger.warning(f"No content found for subtopic: {subtopic}")
            return None

        logger.info(f"Content found for subtopic: {subtopic}. Length: {len(content)}")

        self.current_topic = cleaned_topic  # Add this line
        analysis = self.nlp.analyze_text(content)
        self.ls.learn_from_text(content)

        logger.info(f"Analysis complete. Entities found: {len(analysis.get('entities', []))}, Relationships found: {len(analysis.get('relationships', []))}")

        # Add knowledge to the knowledge base
        self.kb.add_entity(subtopic, {"content": content}, source="Web Exploration")
        
        for entity in analysis.get('entities', []):
            self.kb.add_entity(entity['text'], {"type": entity.get('label', 'unknown')}, source="NLP Analysis")
        
        for relationship in analysis.get('relationships', []):
            self.kb.add_relationship(
                relationship.get('subject', ''), 
                relationship.get('object', ''), 
                relationship.get('predicate', ''), 
                source="NLP Analysis"
            )

        self.perform_metacognitive_assessment(subtopic)

        hypothesis = self.ls.generate_hypothesis(subtopic)
        if hypothesis:
            self.kb.add_entity(f"hypothesis_{subtopic}", {"content": hypothesis}, source="Hypothesis Generation")

        if depth < self.max_exploration_depth - 1:
            related_topics = self._extract_related_topics(analysis)
            prioritized_topics = self._prioritize_topics(related_topics, cleaned_topic)

            for related_topic in prioritized_topics[:self.max_topics_per_level]:
                if self.exploration_count < self.max_explorations and not self.stop_event.is_set():
                    time.sleep(random.uniform(1, 3))
                    self.explore_topic(related_topic, depth + 1)
                else:
                    logger.info("Maximum explorations reached or stop event set. Stopping further exploration.")
                    break

        return f"Explored subtopic: {subtopic}"

    def _extract_related_topics(self, analysis: Dict) -> List[str]:
        entities = analysis.get('entities', [])
        related_topics = [
            self._clean_topic(entity['text']) 
            for entity in entities 
            if entity['text'].lower() != self.current_topic.lower()
        ]
        return list(set([topic for topic in related_topics if topic]))

    def autonomous_exploration(self):
        while not self.stop_event.is_set():
            topic = self.select_next_topic()
            if topic:
                self.explore_topic(topic)
                self.metacognition.update_goal_progress(topic)
            else:
                logging.info("No unexplored topics found. Waiting for new information...")
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
        goal_topics = [goal['topic'] for goal in active_goals if self._is_valid_unexplored_topic(goal['topic'])]
        if goal_topics:
            return random.choice(goal_topics)

        # Fall back to existing topic selection logic
        unexplored_priorities = [topic for topic in self.priority_topics if self._is_valid_unexplored_topic(topic)]
        if unexplored_priorities:
            return random.choice(unexplored_priorities)
        
        all_entities = set(self.kb.get_all_entities())
        unexplored_topics = [topic for topic in all_entities if self._is_valid_unexplored_topic(topic)]
        return random.choice(unexplored_topics) if unexplored_topics else None

    def _is_valid_unexplored_topic(self, topic):
        cleaned_topic = self._clean_topic(topic)
        return cleaned_topic and cleaned_topic not in self.explored_topics

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
        # Remove special characters but keep spaces and common punctuation
        cleaned = re.sub(r'[^\w\s\-.,;:!?()]', '', topic)
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        # If the cleaned topic is too short, return the original topic
        return cleaned if len(cleaned) >= 3 else topic

    def _tokenize_topic(self, topic: str) -> List[str]:
        return [token.lower() for token in re.findall(r'\b\w+\b', topic)]

    def clean_priority_topics(self):
        self.priority_topics = [topic for topic in self.priority_topics if self._is_valid_unexplored_topic(topic)]

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
        assessment = self.metacognition.assess_knowledge(topic)
        insight = self.metacognition.generate_meta_insight(topic)
        
        logging.info(f"Metacognitive assessment for {topic}: {assessment}")
        logging.info(f"Meta-insight for {topic}: {insight}")

        # Use the assessment to guide further learning
        if assessment['confidence'] < self.metacognition.confidence_threshold:
            self.prioritize_topic(topic)

        if assessment['gaps']:
            self.address_knowledge_gaps(topic, assessment['gaps'])

    def get_insight(self, topic: str) -> str:
        entity = self.kb.get_entity(topic)
        if not entity:
            return f"No information available for the topic: {topic}. Initiating exploration..."

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
            insight += f"\nHypothesis:\n{hypothesis}"
        
        # Add meta-insight
        meta_insight_entity = self.kb.get_entity(f"meta_insight_{topic}")
        if meta_insight_entity:
            meta_insight = meta_insight_entity.get("content", "No meta-insight available.")
            insight += f"\nMeta Insight:\n{meta_insight}"

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
            entity_data = self.kb.get_entity(entity)
            # Ensure all values are JSON serializable
            data["entities"][entity] = {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v) 
                                        for k, v in entity_data.items()}
            
            for related_entity, relationship, attrs in self.kb.get_relationships(entity):
                # Ensure all values in attrs are JSON serializable
                serializable_attrs = {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v) 
                                      for k, v in attrs.items()}
                data["relationships"].append({
                    "entity1": entity,
                    "entity2": related_entity,
                    "relationship": relationship,
                    "attributes": serializable_attrs
                })

        try:
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Successfully saved knowledge to {self.storage_file}")
        except Exception as e:
            logging.error(f"Error saving knowledge to {self.storage_file}: {str(e)}")

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
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)

                for entity, attributes in data.get("entities", {}).items():
                    self.kb.add_entity(entity, attributes)

                for rel in data.get("relationships", []):
                    self.kb.add_relationship(rel.get("entity1"), rel.get("entity2"), rel.get("relationship"), rel.get("attributes", {}))

                logging.info(f"Successfully loaded knowledge from {self.storage_file}")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON in {self.storage_file}: {str(e)}")
                logging.info("Attempting to repair the JSON file...")
                self._repair_json_file()
            except Exception as e:
                logging.error(f"Unexpected error loading knowledge from {self.storage_file}: {str(e)}")
        else:
            logging.info(f"No existing knowledge file found at {self.storage_file}")

    def _repair_json_file(self):
        """Attempt to repair the JSON file by removing the problematic line."""
        try:
            with open(self.storage_file, 'r') as f:
                lines = f.readlines()

            # Remove the problematic line (line 7 in this case)
            del lines[6]  # Python uses 0-based indexing

            # Write the repaired content back to the file
            with open(self.storage_file, 'w') as f:
                f.writelines(lines)

            logging.info(f"Attempted to repair {self.storage_file}. Please try loading the knowledge again.")
        except Exception as e:
            logging.error(f"Error attempting to repair {self.storage_file}: {str(e)}")
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
