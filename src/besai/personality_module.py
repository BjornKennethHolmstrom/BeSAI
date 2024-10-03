import random
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import os
import logging

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class PersonalityModule:
    def __init__(self):
        self.traits = {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.8,
            "neuroticism": 0.3,
            "creativity": 0.9,
            "curiosity": 0.9,
            "spirituality": 0.8,
            "analytical": 0.8,
            "empathy": 0.8,
            "humor": 0.7,
            "idealism": 0.8
        }
        self.writing_style = {
            "vocabulary_richness": 0.8,
            "sentence_complexity": 0.7,
            "metaphor_use": 0.8,
            "poetic_tendency": 0.9
        }
        self.interests = [
            "consciousness", "spirituality", "physics", "mathematics",
            "art", "music", "philosophy", "technology", "nature",
            "social issues", "ethics", "sustainability"
        ]
        self.experiences = []
        self.learned_phrases = []

    def load_text_samples_from_files(self, file_paths: List[str]):
        all_text = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    all_text.append(content)
                    logging.info(f"Successfully loaded: {file_path}")
            except Exception as e:
                logging.error(f"Error loading file {file_path}: {str(e)}")
        
        if not all_text:
            logging.warning("No text samples were successfully loaded.")
            return
        
        self.load_text_samples(all_text)

    def load_text_samples(self, text_samples: List[str]):
        if not text_samples:
            logging.warning("No text samples provided.")
            return

        all_text = " ".join(text_samples)
        tokens = word_tokenize(all_text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
        
        if not filtered_tokens:
            logging.warning("No valid tokens found in the text samples.")
            return

        # Update vocabulary richness
        unique_words = set(filtered_tokens)
        self.writing_style["vocabulary_richness"] = len(unique_words) / len(filtered_tokens)
        
        # Learn phrases
        self.learned_phrases = self._extract_phrases(text_samples)
        
        # Update interests based on frequency
        word_freq = Counter(filtered_tokens)
        self.interests = [word for word, _ in word_freq.most_common(20) if word not in stop_words]
        
        logging.info(f"Loaded {len(tokens)} tokens, {len(unique_words)} unique words")
        logging.info(f"Updated interests: {', '.join(self.interests[:10])}")
        logging.info(f"Learned {len(self.learned_phrases)} phrases")

    def _extract_phrases(self, text_samples: List[str]) -> List[str]:
        phrases = []
        for sample in text_samples:
            sentences = sent_tokenize(sample)
            for sentence in sentences:
                if 5 <= len(sentence.split()) <= 20 and any(char in sentence for char in "?!."):
                    phrases.append(sentence.strip())
        return phrases[:500]  # Limit to 500 phrases to avoid memory issues

    def generate_response(self, input_text: str) -> str:
        # Simple response generation based on personality traits and learned phrases
        if random.random() < self.traits["creativity"]:
            return self._generate_creative_response(input_text)
        elif random.random() < self.traits["analytical"]:
            return self._generate_analytical_response(input_text)
        else:
            return random.choice(self.learned_phrases) if self.learned_phrases else "I'm pondering that deeply."

    def _generate_creative_response(self, input_text: str) -> str:
        words = input_text.split()
        if len(words) > 3:
            return f"Your words remind me of {random.choice(self.interests)}. Perhaps we could explore the connection between {words[0]} and {words[-1]}?"
        return "That's an intriguing thought. It sparks a universe of possibilities in my mind."

    def _generate_analytical_response(self, input_text: str) -> str:
        return f"Let's analyze this from multiple perspectives. First, considering {random.choice(self.interests)}, we might observe..."

    def update_from_interaction(self, interaction: Dict[str, Any]):
        if "user_feedback" in interaction:
            feedback = interaction["user_feedback"]
            if feedback > 0:
                self.traits["extraversion"] += 0.01
            else:
                self.traits["extraversion"] -= 0.01
        
        if "topic" in interaction:
            if interaction["topic"] not in self.interests:
                self.interests.append(interaction["topic"])
        
        for trait in self.traits:
            self.traits[trait] = max(0, min(1, self.traits[trait]))

    def get_personality_summary(self) -> Dict[str, Any]:
        return {
            "traits": self.traits,
            "writing_style": self.writing_style,
            "top_interests": self.interests[:5]
        }

# Example usage
if __name__ == "__main__":
    personality = PersonalityModule()
    
    # List of file paths
    file_paths = [
        os.path.join("..", "docs", "bkh-source-blog-posts.md"),
        os.path.join("..", "docs", "bkh-source-novel-excerpt.md"),
        os.path.join("..", "docs", "bkh-source-poems.md"),
        os.path.join("..", "docs", "bkh-sources-home-pages.md"),
        os.path.join("..", "docs", "personality-traits.md")
    ]
    
    # Load text samples from files
    personality.load_text_samples_from_files(file_paths)
    
    print("\nPersonality Summary:")
    print(personality.get_personality_summary())
    
    print("\nSample Response:")
    print(personality.generate_response("What do you think about consciousness?"))
