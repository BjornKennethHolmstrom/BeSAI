import json
from datetime import datetime

class VersionedKnowledgeBase:
    def __init__(self):
        self.knowledge = {}
        self.version = 1
        self.metadata = {}

    def add_knowledge(self, topic, information, source):
        if topic not in self.knowledge:
            self.knowledge[topic] = []
        
        entry = {
            "information": information,
            "metadata": {
                "source": source,
                "acquisition_date": datetime.now().isoformat(),
                "version": self.version
            }
        }
        self.knowledge[topic].append(entry)

    def get_knowledge(self, topic):
        return self.knowledge.get(topic, [])

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "knowledge": self.knowledge,
                "version": self.version,
                "metadata": self.metadata
            }, f, indent=2)

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.knowledge = data["knowledge"]
            self.version = data["version"]
            self.metadata = data["metadata"]

    def flag_improvement(self, topic, suggestion):
        if "improvement_flags" not in self.metadata:
            self.metadata["improvement_flags"] = {}
        self.metadata["improvement_flags"][topic] = suggestion

# Usage example
kb = VersionedKnowledgeBase()
kb.add_knowledge("AI", "Artificial Intelligence is a branch of computer science...", "Wikipedia")
kb.flag_improvement("AI", "Consider adding more specific subcategories of AI")
kb.save_to_file("knowledge_base_v1.json")
