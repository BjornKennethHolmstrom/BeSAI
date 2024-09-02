# BeSAI/src/core/knowledge_extractor.py

from .natural_language_processing import NaturalLanguageProcessing
from .knowledge_base import KnowledgeBase

class KnowledgeExtractor:
    def __init__(self, nlp: NaturalLanguageProcessing, kb: KnowledgeBase):
        self.nlp = nlp
        self.kb = kb

    def process_text(self, text: str):
        entities = self.nlp.extract_entities(text)
        relationships = self.nlp.extract_relationships(text)
        attributes = self.nlp.extract_attributes(text)

        # Add entities to the knowledge base
        for entity in entities:
            self.kb.add_entity(entity['text'], {"type": entity['label']})

        # Add relationships to the knowledge base
        for rel in relationships:
            self.kb.add_entity(rel['subject'], {})  # Ensure subject exists
            self.kb.add_entity(rel['object'], {})   # Ensure object exists
            self.kb.add_relationship(rel['subject'], rel['object'], rel['predicate'])

        # Add attributes to the knowledge base
        for attr in attributes:
            self.kb.add_entity(attr['entity'], {})  # Ensure entity exists
            self.kb.update_entity(attr['entity'], {attr['attribute']: True})

    def extract_knowledge_from_text(self, text: str):
        self.process_text(text)
        return {
            "entities": self.kb.get_all_entities(),
            "relationships": [
                (entity, rel, related_entity)
                for entity in self.kb.get_all_entities()
                for related_entity, rel, _ in self.kb.get_relationships(entity)
            ]
        }

# Example usage
if __name__ == "__main__":
    nlp = NaturalLanguageProcessing()
    kb = KnowledgeBase()
    extractor = KnowledgeExtractor(nlp, kb)

    text = "BeSAI is an AI project that focuses on ethics. It uses machine learning algorithms to make decisions. The project aims to create a benevolent and spiritually aware artificial intelligence."

    knowledge = extractor.extract_knowledge_from_text(text)
    print("Extracted Entities:", knowledge["entities"])
    print("Extracted Relationships:", knowledge["relationships"])
