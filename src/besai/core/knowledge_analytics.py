import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict
import logging

class KnowledgeAnalytics:
    def __init__(self, versioned_kb):
        self.versioned_kb = versioned_kb
        self.performance_metrics = defaultdict(list)
        self.hypothesis_tracking = defaultdict(list)
        self.insight_quality = defaultdict(list)
        self.knowledge_growth = []
        self.last_analysis = datetime.now()
        self.analysis_interval = timedelta(hours=1)

    def track_hypothesis(self, hypothesis: Dict[str, Any], verification_result: bool = None):
        """Track a hypothesis and its verification result if available."""
        timestamp = datetime.now()
        tracking_data = {
            'timestamp': timestamp,
            'hypothesis': hypothesis,
            'verification_result': verification_result,
            'version': self.versioned_kb.version,
            'certainty': hypothesis.get('certainty', 0.0)
        }
        self.hypothesis_tracking[hypothesis['entity']].append(tracking_data)
        
        if verification_result is not None:
            self._update_hypothesis_accuracy(hypothesis['entity'])

    def track_insight(self, topic: str, insight: str, relevance_score: float, user_feedback: Optional[int] = None):
        """Track insights and their quality metrics."""
        timestamp = datetime.now()
        insight_data = {
            'timestamp': timestamp,
            'insight': insight,
            'relevance_score': relevance_score,
            'user_feedback': user_feedback,
            'version': self.versioned_kb.version
        }
        self.insight_quality[topic].append(insight_data)

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance across various metrics."""
        current_time = datetime.now()
        if (current_time - self.last_analysis) < self.analysis_interval:
            return {}

        self.last_analysis = current_time
        
        knowledge_metrics = self._calculate_knowledge_metrics()
        hypothesis_metrics = self._calculate_hypothesis_metrics()
        insight_metrics = self._calculate_insight_metrics()
        growth_metrics = self._calculate_growth_metrics()

        performance_data = {
            'timestamp': current_time,
            'knowledge_metrics': knowledge_metrics,
            'hypothesis_metrics': hypothesis_metrics,
            'insight_metrics': insight_metrics,
            'growth_metrics': growth_metrics,
            'version': self.versioned_kb.version
        }

        self.performance_metrics[current_time].append(performance_data)
        return performance_data

    def _calculate_knowledge_metrics(self) -> Dict[str, Any]:
        """Calculate metrics related to knowledge base quality and coverage."""
        total_entities = len(self.versioned_kb.graph.nodes())
        total_relationships = len(self.versioned_kb.graph.edges())
        
        # Calculate average certainty
        certainties = []
        for node, data in self.versioned_kb.graph.nodes(data=True):
            metadata = data.get('metadata', {})
            certainty = metadata.get('certainty', 0.0)
            certainties.append(certainty)
        
        avg_certainty = np.mean(certainties) if certainties else 0.0
        
        # Calculate connectivity metrics
        if total_entities > 0:
            avg_relationships_per_entity = total_relationships / total_entities
            density = nx.density(self.versioned_kb.graph)
        else:
            avg_relationships_per_entity = 0
            density = 0

        return {
            'total_entities': total_entities,
            'total_relationships': total_relationships,
            'average_certainty': avg_certainty,
            'avg_relationships_per_entity': avg_relationships_per_entity,
            'graph_density': density
        }

    def _calculate_hypothesis_metrics(self) -> Dict[str, Any]:
        """Calculate metrics related to hypothesis generation and verification."""
        total_hypotheses = sum(len(hypotheses) for hypotheses in self.hypothesis_tracking.values())
        if total_hypotheses == 0:
            return {'total_hypotheses': 0, 'accuracy': 0.0, 'avg_certainty': 0.0}

        verified_hypotheses = sum(
            1 for hypotheses in self.hypothesis_tracking.values()
            for h in hypotheses if h['verification_result'] is not None
        )
        
        correct_hypotheses = sum(
            1 for hypotheses in self.hypothesis_tracking.values()
            for h in hypotheses if h.get('verification_result', False)
        )

        accuracy = correct_hypotheses / verified_hypotheses if verified_hypotheses > 0 else 0.0
        avg_certainty = np.mean([
            h['certainty'] for hypotheses in self.hypothesis_tracking.values()
            for h in hypotheses
        ])

        return {
            'total_hypotheses': total_hypotheses,
            'verified_hypotheses': verified_hypotheses,
            'accuracy': accuracy,
            'avg_certainty': avg_certainty
        }

    def _calculate_insight_metrics(self) -> Dict[str, Any]:
        """Calculate metrics related to insight generation and quality."""
        total_insights = sum(len(insights) for insights in self.insight_quality.values())
        if total_insights == 0:
            return {'total_insights': 0, 'avg_relevance': 0.0, 'avg_user_rating': 0.0}

        relevance_scores = [
            insight['relevance_score']
            for insights in self.insight_quality.values()
            for insight in insights
        ]
        
        user_ratings = [
            insight['user_feedback']
            for insights in self.insight_quality.values()
            for insight in insights
            if insight['user_feedback'] is not None
        ]

        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        avg_user_rating = np.mean(user_ratings) if user_ratings else 0.0

        return {
            'total_insights': total_insights,
            'avg_relevance': avg_relevance,
            'avg_user_rating': avg_user_rating,
            'rated_insights': len(user_ratings)
        }

    def _calculate_growth_metrics(self) -> Dict[str, Any]:
        """Calculate metrics related to knowledge base growth over time."""
        current_snapshot = {
            'timestamp': datetime.now(),
            'total_entities': len(self.versioned_kb.graph.nodes()),
            'total_relationships': len(self.versioned_kb.graph.edges()),
            'version': self.versioned_kb.version
        }
        
        self.knowledge_growth.append(current_snapshot)
        
        if len(self.knowledge_growth) < 2:
            return current_snapshot

        previous_snapshot = self.knowledge_growth[-2]
        time_diff = (current_snapshot['timestamp'] - previous_snapshot['timestamp']).total_seconds() / 3600  # hours
        
        entity_growth_rate = (current_snapshot['total_entities'] - previous_snapshot['total_entities']) / time_diff
        relationship_growth_rate = (current_snapshot['total_relationships'] - previous_snapshot['total_relationships']) / time_diff

        return {
            'current': current_snapshot,
            'entity_growth_rate': entity_growth_rate,  # entities per hour
            'relationship_growth_rate': relationship_growth_rate,  # relationships per hour
            'versions_per_hour': (current_snapshot['version'] - previous_snapshot['version']) / time_diff
        }

    def _update_hypothesis_accuracy(self, entity: str):
        """Update accuracy metrics for hypotheses about a specific entity."""
        hypotheses = self.hypothesis_tracking[entity]
        verified_hypotheses = [h for h in hypotheses if h['verification_result'] is not None]
        
        if not verified_hypotheses:
            return
        
        correct_hypotheses = sum(1 for h in verified_hypotheses if h['verification_result'])
        accuracy = correct_hypotheses / len(verified_hypotheses)
        
        logging.info(f"Updated hypothesis accuracy for {entity}: {accuracy:.2f}")

    def generate_performance_report(self, time_period: str = 'day') -> Dict[str, Any]:
        """Generate a comprehensive performance report for the specified time period."""
        now = datetime.now()
        
        if time_period == 'day':
            start_time = now - timedelta(days=1)
        elif time_period == 'week':
            start_time = now - timedelta(weeks=1)
        elif time_period == 'month':
            start_time = now - timedelta(days=30)
        else:
            raise ValueError("Invalid time period. Use 'day', 'week', or 'month'.")

        relevant_metrics = [
            metrics for timestamp, metrics_list in self.performance_metrics.items()
            for metrics in metrics_list
            if timestamp >= start_time
        ]

        if not relevant_metrics:
            return {}

        report = {
            'time_period': time_period,
            'start_time': start_time,
            'end_time': now,
            'metrics_summary': self._calculate_metrics_summary(relevant_metrics),
            'growth_trends': self._calculate_growth_trends(start_time),
            'hypothesis_performance': self._calculate_hypothesis_performance(start_time),
            'insight_quality_trends': self._calculate_insight_quality_trends(start_time)
        }

        return report

    def _calculate_metrics_summary(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for performance metrics."""
        summary = defaultdict(list)
        
        for metric_data in metrics:
            for category, values in metric_data.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            summary[f"{category}_{key}"].append(value)

        return {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for key, values in summary.items()
        }

    def _calculate_growth_trends(self, start_time: datetime) -> Dict[str, Any]:
        """Calculate knowledge base growth trends."""
        relevant_growth = [
            snapshot for snapshot in self.knowledge_growth
            if snapshot['timestamp'] >= start_time
        ]

        if not relevant_growth:
            return {}

        return {
            'entity_growth': self._calculate_trend(
                [(s['timestamp'], s['total_entities']) for s in relevant_growth]
            ),
            'relationship_growth': self._calculate_trend(
                [(s['timestamp'], s['total_relationships']) for s in relevant_growth]
            )
        }

    def _calculate_trend(self, data_points: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Calculate trend statistics for a series of data points."""
        if not data_points:
            return {}

        values = [point[1] for point in data_points]
        times = [(point[0] - data_points[0][0]).total_seconds() / 3600 for point in data_points]

        if len(values) < 2:
            return {'start_value': values[0], 'end_value': values[0], 'growth_rate': 0}

        growth_rate = (values[-1] - values[0]) / times[-1]  # per hour
        
        return {
            'start_value': values[0],
            'end_value': values[-1],
            'growth_rate': growth_rate,
            'percent_change': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        }

    def export_analytics(self, filename: str):
        """Export analytics data to a JSON file."""
        data = {
            'performance_metrics': {str(k): v for k, v in self.performance_metrics.items()},
            'hypothesis_tracking': self.hypothesis_tracking,
            'insight_quality': self.insight_quality,
            'knowledge_growth': self.knowledge_growth
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def import_analytics(self, filename: str):
        """Import analytics data from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.performance_metrics = defaultdict(list, {datetime.fromisoformat(k): v for k, v in data['performance_metrics'].items()})
        self.hypothesis_tracking = defaultdict(list, data['hypothesis_tracking'])
        self.insight_quality = defaultdict(list, data['insight_quality'])
        self.knowledge_growth = data['knowledge_growth']
