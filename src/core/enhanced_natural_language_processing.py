import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from typing import List, Dict, Any, Tuple
import logging

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

class EnhancedNaturalLanguageProcessing:
    def __init__(self, kb: 'EnhancedKnowledgeBase', re: 'ReasoningEngine'):
        self.kb = kb
        self.re = re
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            logging.warning("Could not load the spaCy model. Falling back to basic NLP.")
            self.use_spacy = False
        self.metaphor_sensitivity = 0.5
        self.creativity_level = 0.5

    def tokenize_words(self, text: str) -> List[str]:
        return word_tokenize(text)

    def tokenize_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def pos_tag(self, tokens: List[str]) -> List[tuple]:
        return pos_tag(tokens)

    def preprocess_text(self, text: str) -> List[List[str]]:
        sentences = self.tokenize_sentences(text)
        tokens = [self.tokenize_words(sentence) for sentence in sentences]
        tokens = [self.remove_stopwords(sentence_tokens) for sentence_tokens in tokens]
        tokens = [self.lemmatize(sentence_tokens) for sentence_tokens in tokens]
        return tokens

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        if self.use_spacy:
            doc = self.nlp(text)
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        else:
            # Fallback to basic NER using NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            entities = [{"text": word, "label": tag} for word, tag in pos_tags if tag.startswith('NN')]

        # Enhance entities with knowledge base information
        for entity in entities:
            kb_entity = self.kb.get_entity(entity['text'])
            if kb_entity:
                entity['kb_info'] = kb_entity
            else:
                # Try to infer information about unknown entities
                hypothesis = self.re.generate_hypothesis(entity['text'])
                if hypothesis:
                    entity['inferred_info'] = hypothesis

        return entities

    def extract_relationships(self, text: str) -> List[Dict[str, str]]:
        if self.use_spacy:
            doc = self.nlp(text)
            relationships = []
            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    subject = token.text
                    verb = token.head.text
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "pobj"):
                            object = child.text
                            relationships.append({
                                "subject": subject,
                                "predicate": verb,
                                "object": object
                            })
        else:
            # Fallback to a simple subject-verb-object extraction
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            relationships = []
            for i in range(len(pos_tags) - 2):
                if pos_tags[i][1].startswith('NN') and pos_tags[i+1][1].startswith('VB') and pos_tags[i+2][1].startswith('NN'):
                    relationships.append({
                        "subject": pos_tags[i][0],
                        "predicate": pos_tags[i+1][0],
                        "object": pos_tags[i+2][0]
                    })

        # Enhance relationships with knowledge base information
        for rel in relationships:
            kb_subject = self.kb.get_entity(rel['subject'])
            kb_object = self.kb.get_entity(rel['object'])
            if kb_subject and kb_object:
                rel['kb_info'] = self.kb.get_relationships(rel['subject'])
            else:
                # Try to infer information about unknown relationships
                inferred_rel = self.re.infer_transitive_relationships(rel['subject'], rel['predicate'])
                if inferred_rel:
                    rel['inferred_info'] = inferred_rel

        return relationships

    def extract_attributes(self, text: str) -> List[Dict[str, str]]:
        if self.use_spacy:
            doc = self.nlp(text)
            attributes = []
            for token in doc:
                if token.dep_ == "attr" and token.head.pos_ == "VERB":
                    entity = ""
                    for child in token.head.children:
                        if child.dep_ == "nsubj":
                            entity = child.text
                            break
                    if entity:
                        attributes.append({
                            "entity": entity,
                            "attribute": token.text
                        })
        else:
            # Fallback to a simple noun-adjective pair extraction
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            attributes = []
            for i in range(len(pos_tags) - 1):
                if pos_tags[i][1].startswith('JJ') and pos_tags[i+1][1].startswith('NN'):
                    attributes.append({
                        "entity": pos_tags[i+1][0],
                        "attribute": pos_tags[i][0]
                    })

        # Enhance attributes with knowledge base information
        for attr in attributes:
            kb_entity = self.kb.get_entity(attr['entity'])
            if kb_entity:
                attr['kb_info'] = kb_entity
            else:
                # Try to infer information about unknown attributes
                inferred_attr = self.re.infer_attribute_from_relationships(attr['entity'], attr['attribute'])
                if inferred_attr:
                    attr['inferred_info'] = inferred_attr

        return attributes

    def analyze_text(self, text: str) -> Dict[str, Any]:
        logging.info("Starting text analysis")
        try:
            if not text or not isinstance(text, str):
                logging.warning("Invalid input text for analysis")
                return {}

            preprocessed_text = self.preprocess_text(text)
            pos_tagged_text = [self.pos_tag(sentence_tokens) for sentence_tokens in preprocessed_text]
            
            analysis = {
                'sentence_count': len(preprocessed_text),
                'word_count': sum(len(sentence) for sentence in preprocessed_text),
                'pos_distribution': self._get_pos_distribution(pos_tagged_text),
                'entities': self.extract_entities(text),
                'relationships': self.extract_relationships(text),
                'attributes': self.extract_attributes(text)
            }

            analysis['inferences'] = self._generate_inferences(analysis)
            
            logging.info(f"Text analysis completed. Sentences: {analysis['sentence_count']}, Words: {analysis['word_count']}")
            return analysis

        except Exception as e:
            logging.error(f"Error during text analysis: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'sentence_count': 0,
                'word_count': 0,
                'pos_distribution': {},
                'entities': [],
                'relationships': [],
                'attributes': [],
                'inferences': []
            }

    def _get_pos_distribution(self, pos_tagged_text: List[List[tuple]]) -> Dict[str, int]:
        pos_counts = {}
        for sentence in pos_tagged_text:
            for _, pos in sentence:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        return pos_counts

    def _generate_inferences(self, analysis: Dict[str, Any]) -> List[str]:
        inferences = []
        for entity in analysis['entities']:
            hypothesis = self.re.generate_hypothesis(entity['text'])
            if hypothesis:
                inferences.append(f"Hypothesis for {entity['text']}: {hypothesis}")

        for rel in analysis['relationships']:
            explanation = self.re.explain_inference(rel['subject'], rel['predicate'])
            if explanation:
                inferences.append(explanation)

        return inferences

    def set_metaphor_sensitivity(self, level: float):
        self.metaphor_sensitivity = max(0.0, min(1.0, level))
        
    def set_creativity_level(self, level: float):
        self.creativity_level = max(0.0, min(1.0, level))
        
    def reset_parameters(self):
        self.metaphor_sensitivity = 0.5
        self.creativity_level = 0.5

# Example usage
if __name__ == "__main__":
    from enhanced_knowledge_base import EnhancedKnowledgeBase
    from reasoning_engine import ReasoningEngine

    kb = EnhancedKnowledgeBase()
    re = ReasoningEngine(kb)
    nlp = EnhancedNaturalLanguageProcessing(kb, re)
    
    # Add some initial knowledge
    kb.add_entity("Python", {"type": "programming_language", "paradigm": "multi-paradigm"}, entity_type="Language")
    kb.add_entity("AI", {"type": "field", "description": "Artificial Intelligence"}, entity_type="Field")
    kb.add_relationship("Python", "AI", "used_in", {"strength": "high"})

    text = "Python is widely used in AI and machine learning projects. It's known for its simplicity and versatility."
    
    analysis = nlp.analyze_text(text)
    print("Text Analysis:")
    print(f"Sentence Count: {analysis['sentence_count']}")
    print(f"Word Count: {analysis['word_count']}")
    print(f"POS Distribution: {analysis['pos_distribution']}")
    print(f"Entities: {analysis['entities']}")
    print(f"Relationships: {analysis['relationships']}")
    print(f"Attributes: {analysis['attributes']}")
    print(f"Inferences: {analysis['inferences']}")
