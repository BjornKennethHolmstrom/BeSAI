import threading
import logging
import time
from besai_grpc_client import BeSAIGRPCClient
from besai_kafka_client import BeSAIKafkaClient
from core.enhanced_knowledge_base import EnhancedKnowledgeBase
from core.reasoning_engine import ReasoningEngine
from core.enhanced_natural_language_processing import EnhancedNaturalLanguageProcessing
from core.versioned_knowledge_base import VersionedKnowledgeBase

logger = logging.getLogger(__name__)

class BeSAISystem:
    def __init__(self, grpc_address, kafka_bootstrap_servers):
        self.grpc_client = BeSAIGRPCClient(grpc_address)
        self.kafka_client = BeSAIKafkaClient()
        self.knowledge_base = EnhancedKnowledgeBase()
        self.versioned_kb = VersionedKnowledgeBase()
        self.reasoning_engine = ReasoningEngine(self.knowledge_base)
        self.nlp = EnhancedNaturalLanguageProcessing(self.knowledge_base, self.reasoning_engine)
        self.save_interval = 300  # Save every 5 minutes
        self.running = True
        self.save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.save_thread.start()

    def process_input(self, input_text):
        try:
            output = self.grpc_client.process_input(input_text)
            analysis = self.nlp.analyze_text(input_text)
            cognitive_state = self._extract_cognitive_state(analysis)
            self.kafka_client.send_cognitive_update(cognitive_state)
            
            hypothesis = self.reasoning_engine.generate_hypothesis(input_text)
            if hypothesis:
                self._update_knowledge_base(hypothesis)
            
            # Apply any altered state effects
            output = self.reasoning_engine._apply_psychedelic_effects(output)
            
            return output, analysis, hypothesis
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"An error occurred while processing your input: {str(e)}", None, None

    def explore_topic(self, topic):
        try:
            result = self.grpc_client.explore_grpc(topic)
            analysis = self.nlp.analyze_text(result)
            knowledge_graph = self._extract_knowledge_graph(analysis)
            self.kafka_client.send_knowledge_transfer(knowledge_graph)
            
            insights = self.reasoning_engine.generate_insight(topic)
            
            # Apply any altered state effects
            insights = self.reasoning_engine._apply_psychedelic_effects(insights)
            
            return result, analysis, insights
        except Exception as e:
            logger.error(f"Error exploring topic: {e}")
            return f"An error occurred while exploring the topic: {str(e)}", None, None

    def query_knowledge_base(self, query):
        try:
            results = self.knowledge_base.query(query)
            return results
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return f"An error occurred while querying the knowledge base: {str(e)}"

    def set_altered_state(self, state):
        try:
            self.reasoning_engine.set_altered_state(state)
            return f"Altered state set to: {state}"
        except ValueError as e:
            return str(e)

    def _extract_cognitive_state(self, analysis):
        return {
            "attention_focus": analysis.get('entities', [{}])[0].get('text', '') if analysis.get('entities') else '',
            "emotional_state": 'neutral',  # You might want to implement sentiment analysis here
            "reasoning_depth": analysis.get('word_count', 0),
        }

    def _extract_knowledge_graph(self, analysis):
        return {
            "entities": analysis.get('entities', []),
            "relationships": analysis.get('relationships', []),
            "attributes": analysis.get('attributes', [])
        }

    def _update_knowledge_base(self, hypothesis):
        entity = hypothesis['entity']
        self.knowledge_base.add_entity(entity, hypothesis['known_attributes'])
        for attr, value in hypothesis['inferred_attributes'].items():
            self.knowledge_base.update_entity(entity, {attr: value['value']})
        for suggestion in hypothesis['potential_relationships']:
            self.knowledge_base.add_relationship(suggestion['entity1'], suggestion['entity2'], suggestion['suggested_relationship'])

        self.versioned_kb.add_knowledge(entity, hypothesis, "BeSAI Reasoning Engine")

    def _periodic_save(self):
        while self.running:
            time.sleep(self.save_interval)
            try:
                self.versioned_kb.save_to_file("besai_knowledge_base.json")
                logger.info("Successfully saved versioned knowledge base to file.")
            except Exception as e:
                logger.error(f"Error saving versioned knowledge base: {e}")

    def close(self):
        self.running = False
        self.save_thread.join()
        self.grpc_client.close()
        self.kafka_client.close()
        # Perform a final save before closing
        try:
            self.versioned_kb.save_to_file("besai_knowledge_base.json")
            logger.info("Final save of versioned knowledge base completed.")
        except Exception as e:
            logger.error(f"Error during final save of versioned knowledge base: {e}")

# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    besai_system = BeSAISystem("localhost:50051", "localhost:9092")
    
    output, analysis, hypothesis = besai_system.process_input("What is the nature of consciousness?")
    print("Output:", output)
    print("Analysis:", analysis)
    print("Hypothesis:", hypothesis)
    
    besai_system.set_altered_state("meditation")
    
    result, analysis, insights = besai_system.explore_topic("artificial intelligence")
    print("Exploration result:", result)
    print("Analysis:", analysis)
    print("Insights:", insights)
    
    query_result = besai_system.query_knowledge_base("consciousness")
    print("Query result:", query_result)
    
    # When you're done, don't forget to close the system
    besai_system.close()
