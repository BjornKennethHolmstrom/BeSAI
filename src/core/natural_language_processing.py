# BeSAI/src/core/natural_language_processing.py

import nltk
nltk.download('punkt')

class NaturalLanguageProcessing:
    def __init__(self):
        pass

    def tokenize(self, text):
        return nltk.word_tokenize(text)
