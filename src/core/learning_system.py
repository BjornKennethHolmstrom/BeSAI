import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.natural_language_processing import NaturalLanguageProcessing
from core.reasoning_engine import ReasoningEngine
from core.enhanced_knowledge_base import EnhancedKnowledgeBase

from typing import List, Dict, Any

class LearningSystem:
    def __init__(self, kb: EnhancedKnowledgeBase, nlp: NaturalLanguageProcessing, re: ReasoningEngine):
        self.kb = kb
        self.nlp = nlp
        self.re = re
        self.learning_history = []

    def learn_from_text(self, text: str):
        # Extract knowledge from text
        analysis = self.nlp.analyze_text(text)
        
        # Add entities and relationships to knowledge base
        for entity in analysis['entities']:
            self.kb.add_entity(entity['text'], {"type": entity['label']}, entity_type=entity['label'], certainty=0.8)
        
        for relationship in analysis['relationships']:
            self.kb.add_relationship(relationship['subject'], relationship['object'], relationship['predicate'], certainty=0.7)
        
        # Add attributes
        for attribute in analysis['attributes']:
            self.kb.update_entity(attribute['entity'], {attribute['attribute']: True}, certainty=0.6)
        
        self.learning_history.append(f"Learned from text: {text[:50]}...")

    def generate_hypothesis(self, entity: str):
        hypothesis = self.re.generate_hypothesis(entity)
        if hypothesis:
            self.learning_history.append(f"Generated hypothesis for: {entity}")
            return hypothesis
        return None

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
