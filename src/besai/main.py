import threading
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from besai_grpc_client import BeSAIGRPCClient
from besai_kafka_client import BeSAIKafkaClient
from core.enhanced_knowledge_base import EnhancedKnowledgeBase
from core.reasoning_engine import ReasoningEngine
from core.enhanced_natural_language_processing import EnhancedNaturalLanguageProcessing
from core.versioned_knowledge_base import VersionedKnowledgeBase
from core.knowledge_analytics import KnowledgeAnalytics
from ethics.ethical_oversight_client import EthicalOversightClient
from ethics.ethical_hooks import EthicalHooks

logger = logging.getLogger(__name__)

class BeSAISystem:
    def __init__(self, grpc_address: str, kafka_bootstrap_servers: str, ethical_oversight_address: str = "localhost:50052"):
        """Initialize BeSAI system with ethical oversight."""
        # Initialize core components
        self.knowledge_base = EnhancedKnowledgeBase()
        self.versioned_kb = VersionedKnowledgeBase()
        self.reasoning_engine = ReasoningEngine(self.knowledge_base)
        self.nlp = EnhancedNaturalLanguageProcessing(self.knowledge_base, self.reasoning_engine)
        self.analytics = KnowledgeAnalytics(self.versioned_kb)

        # Initialize communication clients
        self.grpc_client = BeSAIGRPCClient(grpc_address)
        self.kafka_client = BeSAIKafkaClient()
        
        # Initialize ethical components with analytics
        self.ethical_client = EthicalOversightClient(ethical_oversight_address)
        self.ethical_hooks = EthicalHooks(self.ethical_client)
        self.ethical_analytics = EthicalAnalytics(self.ethical_client.storage)
        
        # Add ethical analysis interval
        self.ethical_analysis_interval = 3600  # 1 hour
        self.ethical_analysis_thread = threading.Thread(target=self._periodic_ethical_analysis, daemon=True)
        self.ethical_analysis_thread.start()

        # System settings
        self.save_interval = 300
        self.metrics_interval = 60
        self.running = True

        # Start background threads
        self.save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.metrics_thread = threading.Thread(target=self._periodic_metrics, daemon=True)
        self.save_thread.start()
        self.metrics_thread.start()

    def process_input(self, input_text: str) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Process input text with ethical oversight."""
        try:
            # First, evaluate the input text itself
            is_approved, reasoning = self.ethical_hooks.evaluate_reasoning_output(
                query=input_text,
                result="",  # Empty result as this is pre-processing
                confidence=1.0
            )
            
            if not is_approved:
                logger.warning(f"Input rejected by ethical oversight: {reasoning}")
                return f"Input rejected: {reasoning}", None, None

            # Process the input if approved
            start_time = time.time()
            analysis = self.nlp.analyze_text(input_text)
            
            # Generate hypothesis
            hypothesis = self.reasoning_engine.generate_hypothesis(input_text)
            
            # Process through gRPC
            output = self.grpc_client.process_input(input_text)

            # Evaluate the output
            is_approved, reasoning = self.ethical_hooks.evaluate_reasoning_output(
                query=input_text,
                result=output,
                confidence=hypothesis.get('certainty', 0.8) if hypothesis else 0.8
            )

            if not is_approved:
                logger.warning(f"Output rejected by ethical oversight: {reasoning}")
                return f"Output rejected: {reasoning}", analysis, None

            # Extract and evaluate cognitive state
            cognitive_state = self._extract_cognitive_state(analysis)
            is_approved, reasoning = self.ethical_hooks.evaluate_cognitive_update(
                cognitive_state=cognitive_state,
                nlp_analysis=analysis
            )

            if is_approved:
                self.kafka_client.send_cognitive_update(cognitive_state, analysis)
            else:
                logger.warning(f"Cognitive update rejected: {reasoning}")

            # Update knowledge base if hypothesis exists and is approved
            if hypothesis:
                is_approved, reasoning = self.ethical_hooks.evaluate_knowledge_update(
                    entity=hypothesis['entity'],
                    attributes=hypothesis['known_attributes'],
                    source='input_processing',
                    certainty=hypothesis.get('certainty', 0.8)
                )

                if is_approved:
                    self._update_knowledge_base(
                        hypothesis['entity'],
                        hypothesis['known_attributes'],
                        'input_processing',
                        hypothesis.get('certainty', 0.8)
                    )
                    self.analytics.track_hypothesis(hypothesis)
                else:
                    logger.warning(f"Knowledge update rejected: {reasoning}")

            # Track performance
            processing_time = time.time() - start_time
            self.analytics.track_insight(input_text, output, self._calculate_relevance_score(analysis))

            return output, analysis, hypothesis

        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            return f"Error processing input: {str(e)}", None, None

    def explore_topic(self, topic: str) -> Tuple[str, Dict[str, Any], str]:
        """Explore a topic with ethical oversight."""
        try:
            # Evaluate the exploration request
            is_approved, reasoning = self.ethical_hooks.evaluate_reasoning_output(
                query=f"explore: {topic}",
                result="",
                confidence=1.0
            )
            
            if not is_approved:
                logger.warning(f"Topic exploration rejected: {reasoning}")
                return f"Topic exploration rejected: {reasoning}", None, None

            # Get results through gRPC
            result = self.grpc_client.explore_grpc(topic)
            
            # Analyze the topic
            analysis = self.nlp.analyze_text(topic)
            
            # Generate insights
            insights = self.reasoning_engine.generate_insight(topic)

            # Evaluate the results
            is_approved, reasoning = self.ethical_hooks.evaluate_reasoning_output(
                query=topic,
                result=f"{result}\n{insights}",
                confidence=0.8
            )

            if not is_approved:
                logger.warning(f"Exploration results rejected: {reasoning}")
                return f"Results rejected: {reasoning}", analysis, None

            # Extract and send knowledge graph if approved
            knowledge_graph = self._extract_knowledge_graph(analysis)
            is_approved, reasoning = self.ethical_hooks.evaluate_knowledge_update(
                entity=topic,
                attributes=knowledge_graph,
                source='topic_exploration',
                certainty=0.8
            )

            if is_approved:
                self.kafka_client.send_knowledge_transfer(knowledge_graph)
            else:
                logger.warning(f"Knowledge transfer rejected: {reasoning}")

            # Track the exploration
            self.analytics.track_insight(topic, insights, self._calculate_relevance_score(analysis))

            return result, analysis, insights

        except Exception as e:
            logger.error(f"Error exploring topic: {str(e)}", exc_info=True)
            return f"Error exploring topic: {str(e)}", None, None

    def query_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Query the knowledge base and return results."""
        try:
            results = self.knowledge_base.query({"text": query})
            self.analytics.track_query(query, bool(results))
            return results or {}
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}", exc_info=True)
            return {}

    def set_altered_state(self, state: str) -> str:
        """Set the system's altered state for reasoning."""
        try:
            self.reasoning_engine.set_altered_state(state)
            return f"Successfully set altered state to: {state}"
        except ValueError as e:
            return str(e)

    def _update_knowledge_base(self, entity: str, attributes: Dict[str, Any], source: str, certainty: float):
        """Update knowledge base with conflict resolution."""
        try:
            # Check for conflicts
            existing_attrs = self.knowledge_base.get_entity(entity)
            if existing_attrs:
                conflicts = self._detect_conflicts(existing_attrs, attributes)
                if conflicts:
                    resolution_details = self._resolve_conflicts(entity, conflicts, attributes)
                    self.kafka_client.send_conflict_resolution_notification(
                        entity=entity,
                        conflict_type="attribute",
                        resolution_strategy=resolution_details["strategy"],
                        resolution_details=resolution_details
                    )

            # Update knowledge base
            self.knowledge_base.add_entity(entity, attributes, source, certainty)
            
            # Send update notification
            self.kafka_client.send_knowledge_update_notification(
                update_type="modification" if existing_attrs else "addition",
                affected_entities=[entity],
                changes={
                    "source": source,
                    "certainty": certainty,
                    "modifications": {entity: attributes}
                },
                version_info={
                    "previous_version": self.versioned_kb.version,
                    "new_version": self.versioned_kb.version + 1,
                    "update_type": "incremental"
                }
            )
            
            self.versioned_kb.increment_version()

        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}", exc_info=True)
            raise

    def _extract_cognitive_state(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cognitive state from analysis results."""
        return {
            "attention_focus": analysis.get('entities', [{}])[0].get('text', '') if analysis.get('entities') else '',
            "emotional_state": analysis.get('sentiment', 'neutral'),
            "reasoning_depth": analysis.get('word_count', 0),
            "state_params": self.reasoning_engine.state_params,
            "focus_level": self.reasoning_engine.focus_level,
            "associative_thinking": self.reasoning_engine.associative_thinking
        }

    def _extract_knowledge_graph(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge graph from analysis results."""
        return {
            "entities": analysis.get('entities', []),
            "relationships": analysis.get('relationships', []),
            "attributes": analysis.get('attributes', [])
        }

    def _detect_conflicts(self, existing_attrs: Dict[str, Any], new_attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Detect conflicts between existing and new attributes."""
        conflicts = {}
        for key, new_value in new_attrs.items():
            if key in existing_attrs and existing_attrs[key] != new_value:
                conflicts[key] = {
                    "original_value": existing_attrs[key],
                    "conflicting_value": new_value
                }
        return conflicts

    def _resolve_conflicts(self, entity: str, conflicts: Dict[str, Any], new_attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts using configured strategy."""
        resolution_details = {
            "entity": entity,
            "strategy": "merge",
            "resolutions": {}
        }
        
        for attr, conflict in conflicts.items():
            if isinstance(conflict["original_value"], list):
                resolved_value = conflict["original_value"] + [conflict["conflicting_value"]]
            else:
                resolved_value = [conflict["original_value"], conflict["conflicting_value"]]
            
            resolution_details["resolutions"][attr] = {
                "original_value": conflict["original_value"],
                "conflicting_value": conflict["conflicting_value"],
                "resolved_value": resolved_value,
                "confidence": 0.8
            }
            
            new_attrs[attr] = resolved_value
        
        return resolution_details

    def _calculate_relevance_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate relevance score for insights."""
        entity_count = len(analysis.get('entities', []))
        relationship_count = len(analysis.get('relationships', []))
        attribute_count = len(analysis.get('attributes', []))
        
        # Simple scoring based on content richness
        base_score = (entity_count + relationship_count + attribute_count) / 10
        return min(1.0, base_score)

    def _periodic_save(self):
        """Periodically save knowledge base and analytics data."""
        while self.running:
            try:
                self.versioned_kb.save_to_file("besai_knowledge_base.json")
                self.analytics.export_analytics("besai_analytics.json")
                logger.info("Successfully saved system state")
            except Exception as e:
                logger.error(f"Error during periodic save: {str(e)}", exc_info=True)
            time.sleep(self.save_interval)

    def _periodic_metrics(self):
        """Periodically send system metrics."""
        while self.running:
            try:
                metrics = self._collect_system_metrics()
                performance_data = self._collect_performance_data()
                
                self.kafka_client.send_system_metrics(
                    metrics=metrics,
                    performance_data=performance_data
                )
                
                # Log metrics summary
                logger.info(
                    f"System metrics - KB size: {metrics['knowledge_base_size']}, "
                    f"Pending retries: {metrics['message_retry_status']['pending_retries']}, "
                    f"DLQ size: {metrics['message_retry_status']['dead_letter_queue']}"
                )
                
            except Exception as e:
                logger.error(f"Error sending system metrics: {str(e)}")
            
            time.sleep(self.metrics_interval)

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics including retry status."""
        try:
            metrics = {
                "knowledge_base_size": len(self.knowledge_base.graph.nodes()),
                "relationship_count": len(self.knowledge_base.graph.edges()),
                "entity_types_count": len(self.knowledge_base.entity_types),
                "average_certainty": self._calculate_average_certainty(),
                "version": self.versioned_kb.version,
                
                # Add retry metrics
                "message_retry_status": {
                    "pending_retries": len(list(self.kafka_client.retry_processor.failed_messages_path.glob("*.json"))),
                    "dead_letter_queue": len(list(self.kafka_client.retry_processor.dlq_path.glob("*.json"))),
                    "active_retries": self.kafka_client.retry_processor.processing_queue.qsize(),
                    "retry_processor_status": "running" if self.kafka_client.retry_processor.running else "stopped"
                }
            }
            
            # Get detailed retry status
            retry_status = self.kafka_client.retry_processor.get_retry_status()
            metrics["message_retry_details"] = retry_status
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {}

    def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect system performance data."""
        process = psutil.Process()
        return {
            "processing_time": self.analytics.get_average_processing_time(),
            "memory_usage": f"{process.memory_info().rss / 1024 / 1024:.2f}MB",
            "cpu_percent": process.cpu_percent(),
            "thread_count": process.num_threads(),
            "success_rate": self.analytics.get_success_rate()
        }

    def _calculate_average_certainty(self) -> float:
        """Calculate average certainty across all knowledge base entries."""
        try:
            certainties = []
            for node, data in self.knowledge_base.graph.nodes(data=True):
                metadata = data.get('metadata', {})
                certainty = metadata.get('certainty', 0.0)
                certainties.append(certainty)
            
            return sum(certainties) / len(certainties) if certainties else 0.0
        except Exception as e:
            logger.error(f"Error calculating average certainty: {str(e)}", exc_info=True)
            return 0.0

def _periodic_ethical_analysis(self):
    """Periodically run ethical analysis."""
    while self.running:
        try:
            # Generate analysis report
            analysis_report = self.ethical_analytics.generate_analysis_report()
            risk_metrics = self.ethical_analytics.calculate_risk_metrics()
            
            # Send analytics through Kafka
            self.kafka_client.send_system_metrics(
                metrics={
                    "ethical_analysis": analysis_report,
                    "ethical_risk_metrics": risk_metrics
                },
                performance_data=self._collect_ethical_performance_data()
            )
            
            # Log important findings
            if risk_metrics.get("risk_score", 0) > 70:  # High risk threshold
                logger.warning(f"High ethical risk score detected: {risk_metrics['risk_score']}")
                for action in risk_metrics.get("high_risk_actions", []):
                    logger.warning(f"High risk action: {action}")
            
            # Export analysis if significant findings
            if self._should_export_analysis(analysis_report, risk_metrics):
                export_path = self.ethical_analytics.export_analysis()
                logger.info(f"Exported ethical analysis to: {export_path}")
            
        except Exception as e:
            logger.error(f"Error in ethical analysis: {str(e)}")
        
        time.sleep(self.ethical_analysis_interval)

    def _collect_ethical_performance_data(self) -> Dict[str, Any]:
        """Collect ethical oversight performance metrics."""
        try:
            return {
                "ethical_decisions_total": len(self.ethical_client.get_decision_history()),
                "ethical_decision_rate": self._calculate_decision_rate(),
                "ethical_service_latency": self._measure_ethical_service_latency(),
                "ethical_cache_hit_rate": self._calculate_cache_hit_rate()
            }
        except Exception as e:
            logger.error(f"Error collecting ethical performance data: {str(e)}")
            return {}

    def _calculate_decision_rate(self) -> float:
        """Calculate the rate of ethical decisions per minute."""
        try:
            recent_decisions = self.ethical_client.get_decision_history(limit=100)
            if not recent_decisions:
                return 0.0
            
            time_diff = (datetime.now() - datetime.fromisoformat(recent_decisions[0].timestamp)).total_seconds()
            return len(recent_decisions) / (time_diff / 60) if time_diff > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating decision rate: {str(e)}")
            return 0.0

    def _measure_ethical_service_latency(self) -> float:
        """Measure the latency of ethical service calls."""
        try:
            start_time = time.time()
            self.ethical_client.get_ethical_guidelines(force_refresh=True)
            return time.time() - start_time
        except Exception as e:
            logger.error(f"Error measuring ethical service latency: {str(e)}")
            return -1.0

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate the cache hit rate for ethical guidelines."""
        try:
            cache_stats = self.ethical_client.storage.get_cached_guidelines()
            if not cache_stats:
                return 0.0
            return cache_stats.get("hit_rate", 0.0)
        except Exception as e:
            logger.error(f"Error calculating cache hit rate: {str(e)}")
            return 0.0

    def _should_export_analysis(self, analysis_report: Dict[str, Any], risk_metrics: Dict[str, Any]) -> bool:
        """Determine if analysis should be exported based on findings."""
        try:
            # Export if any of these conditions are met
            conditions = [
                risk_metrics.get("risk_score", 0) > 70,  # High risk score
                len(risk_metrics.get("high_risk_actions", [])) > 2,  # Multiple high-risk actions
                analysis_report.get("summary_metrics", {}).get("approval_rate", 100) < 80,  # Low approval rate
                analysis_report.get("trend_analysis", {}).get("trend_direction", {}).get("approval_rate") == "decreasing"
            ]
            return any(conditions)
        except Exception as e:
            logger.error(f"Error in export decision: {str(e)}")
            return False

    def close(self):
        """Clean up system resources including ethical oversight."""
        try:
            self.running = False
            
            # Wait for background threads
            self.save_thread.join()
            self.metrics_thread.join()
            self.ethical_analysis_thread.join()
            
            # Perform final analysis export
            try:
                self.ethical_analytics.export_analysis()
            except Exception as e:
                logger.error(f"Error in final analysis export: {str(e)}")
            
            # Perform final save
            self.versioned_kb.save_to_file("besai_knowledge_base.json")
            self.analytics.export_analytics("besai_analytics.json")
            
            # Close clients
            self.grpc_client.close()
            self.kafka_client.close()
            self.ethical_client.close()
            
            logger.info("Successfully closed BeSAI system")
        except Exception as e:
            logger.error(f"Error during system shutdown: {str(e)}", exc_info=True)

# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    system = BeSAISystem("localhost:50051", "localhost:9092")
    
    try:
        # Process some input
        output, analysis, hypothesis = system.process_input("What is the nature of consciousness?")
        print("Output:", output)
        print("Analysis:", analysis)
        print("Hypothesis:", hypothesis)
        
        # Explore a topic
        result, analysis, insights = system.explore_topic("artificial intelligence")
        print("Exploration result:", result)
        print("Analysis:", analysis)
        print("Insights:", insights)
        
        # Query the knowledge base
        query_result = system.query_knowledge_base("consciousness")
        print("Query result:", query_result)
        
        # Try different altered states
        system.set_altered_state("meditation")
        output, analysis, hypothesis = system.process_input("What is the meaning of life?")
        print("Meditation state output:", output)
        
    finally:
        # Ensure proper cleanup
        system.close()
