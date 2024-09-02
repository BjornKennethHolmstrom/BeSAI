# BeSAI/src/core/natural_language_processing.py

import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from typing import List, Dict, Any
import logging

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

class NaturalLanguageProcessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            logging.warning("Could not load the spaCy model. Falling back to basic NLP.")
            self.use_spacy = False

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
            return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        else:
            # Fallback to basic NER using NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            return [{"text": word, "label": tag} for word, tag in pos_tags if tag.startswith('NN')]

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
            return relationships
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
            return attributes
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
            return attributes

    def analyze_text(self, text: str) -> Dict[str, Any]:
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
        
        return analysis

    def _get_pos_distribution(self, pos_tagged_text: List[List[tuple]]) -> Dict[str, int]:
        pos_counts = {}
        for sentence in pos_tagged_text:
            for _, pos in sentence:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        return pos_counts

# Example usage
if __name__ == "__main__":
    nlp = NaturalLanguageProcessing()
    
    text = "BeSAI is an AI project that focuses on ethics. It uses machine learning algorithms to make decisions."
    
    analysis = nlp.analyze_text(text)
    print("Text Analysis:")
    print(f"Sentence Count: {analysis['sentence_count']}")
    print(f"Word Count: {analysis['word_count']}")
    print(f"POS Distribution: {analysis['pos_distribution']}")
    print(f"Entities: {analysis['entities']}")
    print(f"Relationships: {analysis['relationships']}")
    print(f"Attributes: {analysis['attributes']}")
