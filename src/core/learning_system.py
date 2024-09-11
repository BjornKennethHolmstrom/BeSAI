import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.natural_language_processing import NaturalLanguageProcessing
from core.reasoning_engine import ReasoningEngine
from core.enhanced_knowledge_base import EnhancedKnowledgeBase

from typing import List, Dict, Any, Tuple, Optional

import logging

class LearningSystem:
    def __init__(self, knowledgebase, nlp, reasoning_engine, metacognition):
        self.kb = knowledgebase
        self.nlp = nlp
        self.re = reasoning_engine
        self.metacognition = metacognition
        self.learning_history = []

    def generate_hypothesis(self, topic: str) -> Optional[str]:
        logging.info(f"Generating hypothesis for topic: {topic}")
        try:
            # Get the entity and its relationships
            entity = self.kb.get_entity(topic)
            if not entity:
                logging.warning(f"No entity found for topic: {topic}")
                return None

            relationships = self.kb.get_relationships(topic)
            
            # Get metacognitive assessment
            assessment = self.metacognition.assess_knowledge(topic)
            
            # Generate base hypothesis
            base_hypothesis = self._generate_base_hypothesis(topic, entity, relationships)
            
            # Enhance hypothesis with cross-domain insights
            enhanced_hypothesis = self._enhance_with_cross_domain_insights(base_hypothesis, topic)
            
            # Add metacognitive reflection
            final_hypothesis = self._add_metacognitive_reflection(enhanced_hypothesis, assessment)
            
            logging.info(f"Successfully generated hypothesis for {topic}")
            return final_hypothesis

        except Exception as e:
            logging.exception(f"Error generating hypothesis for {topic}: {str(e)}")
            return None

    def _generate_base_hypothesis(self, topic: str, entity: Dict[str, Any], relationships: List[Tuple[str, str, Dict[str, Any]]]) -> str:
        hypothesis = f"Based on the current understanding of {topic}, "
        
        # Add information about attributes
        if entity:
            attributes = [f"{key} is {value}" for key, value in entity.items() if key != 'metadata']
            if attributes:
                hypothesis += "it appears that " + ", ".join(attributes[:3]) + ". "
        
        # Add information about relationships
        if relationships:
            rel_info = [f"it {rel} {related_entity}" for related_entity, rel, _ in relationships[:3]]
            hypothesis += "Furthermore, " + ", and ".join(rel_info) + ". "
        
        return hypothesis

    def _enhance_with_cross_domain_insights(self, base_hypothesis: str, topic: str) -> str:
        # Get related topics from different domains
        related_topics = self.kb.get_related_topics(topic, max_distance=2, max_topics=5)
        
        if related_topics:
            insight = "Interestingly, this topic might have connections to "
            topic_insights = []
            for related_topic in related_topics:
                relationship = self.re.infer_relationship(topic, related_topic)
                if relationship:
                    topic_insights.append(f"{related_topic} ({relationship})")
            
            if topic_insights:
                insight += ", ".join(topic_insights) + ". "
                insight += "These cross-domain connections suggest that "
                insight += self._generate_creative_insight(topic, related_topics)
                return base_hypothesis + insight
        
        return base_hypothesis

    def _generate_creative_insight(self, topic: str, related_topics: List[str]) -> str:
        templates = [
            f"there might be underlying principles connecting {topic} to diverse fields of study.",
            f"the concepts in {topic} could potentially be applied to solve problems in {random.choice(related_topics)}.",
            f"studying {topic} from the perspective of {random.choice(related_topics)} might yield novel insights.",
            f"there could be a unifying theory that encompasses both {topic} and {random.choice(related_topics)}.",
        ]
        return random.choice(templates)

    def _add_metacognitive_reflection(self, hypothesis: str, assessment: Dict[str, Any]) -> str:
        reflection = f"\n\nReflecting on this hypothesis, "
        if assessment['confidence'] < 0.5:
            reflection += f"I acknowledge that my understanding of {assessment['topic']} is limited. "
            reflection += "This hypothesis should be considered speculative and requires further investigation. "
        elif assessment['confidence'] < 0.8:
            reflection += f"I have a moderate level of confidence in my understanding of {assessment['topic']}. "
            reflection += "While this hypothesis is grounded in my current knowledge, it may benefit from additional research and validation. "
        else:
            reflection += f"I have a high degree of confidence in my understanding of {assessment['topic']}. "
            reflection += "This hypothesis is well-supported by my current knowledge, but as always, I remain open to new information that might refine or challenge these ideas. "

        if assessment['gaps']:
            reflection += f"Areas that could strengthen this hypothesis include: {', '.join(assessment['gaps'])}. "

        return hypothesis + reflection

    def learn_from_text(self, text: str):
        # Extract knowledge from text
        analysis = self.nlp.analyze_text(text)
        
        # Add entities and relationships to knowledge base
        for entity in analysis['entities']:
            self.kb.add_entity(entity['text'], {"type": entity['label']}, entity_type=entity['label'], source="NLP Analysis", certainty=0.8)
        
        for relationship in analysis['relationships']:
            self.kb.add_relationship(
                relationship['subject'],
                relationship['object'],
                relationship['predicate'],
                source="NLP Analysis",
                certainty=0.7
            )
        
        # Add attributes
        for attribute in analysis['attributes']:
            self.kb.update_entity(attribute['entity'], {attribute['attribute']: attribute['value']}, certainty=0.6)
        
        self.learning_history.append(f"Learned from text: {text[:50]}...")

    def apply_inference(self, start_entity: str, relationship_type: str):
        inferred_relationships = self.re.infer_transitive_relationships(start_entity, relationship_type)
        for rel in inferred_relationships:
            if rel['inferred']:
                self.kb.add_relationship(rel['from'], rel['to'], rel['relationship'], certainty=0.5)
        self.learning_history.append(f"Applied inference for {start_entity} with relationship {relationship_type}")

    def learn_from_hypothesis(self, hypothesis: Dict[str, Any]):
        entity = hypothesis['entity']
        for attr, value in hypothesis['inferred_attributes'].items():
            self.kb.update_entity(entity, {attr: value}, certainty=0.4)
        
        for suggestion in hypothesis['potential_relationships']:
            self.kb.add_relationship(suggestion['entity1'], suggestion['entity2'], suggestion['suggested_relationship'], certainty=0.3)
        
        self.learning_history.append(f"Learned from hypothesis for: {entity}")

    def get_learning_history(self) -> List[str]:
        return self.learning_history

    def suggest_learning_topics(self) -> List[str]:
        all_entities = self.kb.graph.nodes()
        topics = []
        for entity in all_entities:
            if len(self.kb.get_relationships(entity)) < 2:  # Entities with few relationships
                topics.append(f"Learn more about {entity}")
            entity_data = self.kb.get_entity(entity)
            if entity_data:
                certainty = entity_data.get('certainty', 1.0)
                if isinstance(certainty, dict):
                    if any(cert < 0.5 for cert in certainty.values()):
                        topics.append(f"Verify information about {entity}")
                elif certainty < 0.5:
                    topics.append(f"Verify information about {entity}")
        return topics

    def analyze_sentiment(self, text: str) -> float:
        """
        Placeholder method for sentiment analysis.
        Returns a sentiment score between -1 (very negative) and 1 (very positive).
        """
        logging.warning("Using placeholder sentiment analysis. Implement a proper sentiment analysis method for more accurate results.")
        return 0.0  # Neutral sentiment as a placeholder
    
    def categorize_perspective(self, text: str) -> str:
        """
        Placeholder method for perspective categorization.
        Returns a string representing the perspective category.
        """
        logging.warning("Using placeholder perspective categorization. Implement a proper categorization method for more accurate results.")
        return "neutral"  # Neutral perspective as a placeholder

# Example usage
if __name__ == "__main__":
    kb = EnhancedKnowledgeBase()
    nlp = NaturalLanguageProcessing()
    re = ReasoningEngine(kb)
    ls = LearningSystem(kb, nlp, re)

    # Learning from text
    ls.learn_from_text("Python is a popular programming language used in AI and machine learning.")
    
    # Generate and apply hypothesis
    hypothesis = ls.generate_hypothesis("Python")
    if hypothesis:
        ls.learn_from_hypothesis(hypothesis)
    
    # Apply inference
    ls.apply_inference("Python", "used_in")
    
    # Print learning history
    print("Learning History:")
    for entry in ls.get_learning_history():
        print(f"- {entry}")
    
    # Suggest learning topics
    print("\nSuggested Learning Topics:")
    for topic in ls.suggest_learning_topics():
        print(f"- {topic}")

    # Print final knowledge base state
    print("\nFinal Knowledge Base State:")
    for entity in kb.graph.nodes():
        print(f"Entity: {entity}")
        print(f"Attributes: {kb.get_entity(entity)}")
        print(f"Relationships: {kb.get_relationships(entity)}")
        print()
