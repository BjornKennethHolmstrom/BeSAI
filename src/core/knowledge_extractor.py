# BeSAI/src/core/knowledge_extractor.py

from .natural_language_processing import NaturalLanguageProcessing
from .knowledge_base import KnowledgeBase
from typing import Dict, Any

class KnowledgeExtractor:
    def __init__(self, nlp: NaturalLanguageProcessing, kb: KnowledgeBase):
        self.nlp = nlp
        self.kb = kb

    def process_text(self, text: str) -> Dict[str, Any]:
        logging.info(f"Processing text: {text[:50]}...")  # Log first 50 chars
        try:
            entities = self.nlp.extract_entities(text)
            relationships = self.nlp.extract_relationships(text)
            attributes = self.nlp.extract_attributes(text)

            # Add entities to the knowledge base
            for entity in entities:
                try:
                    self.kb.add_entity(entity['text'], {"type": entity.get('label', 'unknown')})
                    logging.debug(f"Added entity: {entity['text']}")
                except Exception as e:
                    logging.error(f"Error adding entity {entity['text']}: {str(e)}")

            # Add relationships to the knowledge base
            for rel in relationships:
                try:
                    self.kb.add_entity(rel.get('subject', ''), {})  # Ensure subject exists
                    self.kb.add_entity(rel.get('object', ''), {})   # Ensure object exists
                    self.kb.add_relationship(rel.get('subject', ''), rel.get('object', ''), rel.get('predicate', ''))
                    logging.debug(f"Added relationship: {rel.get('subject', '')} - {rel.get('predicate', '')} - {rel.get('object', '')}")
                except Exception as e:
                    logging.error(f"Error adding relationship: {str(e)}")

            # Add attributes to the knowledge base
            for attr in attributes:
                try:
                    self.kb.add_entity(attr.get('entity', ''), {})  # Ensure entity exists
                    self.kb.update_entity(attr.get('entity', ''), {attr.get('attribute', ''): True})
                    logging.debug(f"Added attribute: {attr.get('entity', '')} - {attr.get('attribute', '')}")
                except Exception as e:
                    logging.error(f"Error adding attribute: {str(e)}")

            logging.info("Text processing completed successfully")
            return {
                "entities": len(entities),
                "relationships": len(relationships),
                "attributes": len(attributes)
            }

        except Exception as e:
            logging.exception(f"Error processing text: {str(e)}")
            return {
                "entities": 0,
                "relationships": 0,
                "attributes": 0,
                "error": str(e)
            }

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
