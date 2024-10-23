import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import json
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)

class EthicalAnalytics:
    def __init__(self, ethical_storage):
        """
        Initialize the ethical analytics system.
        
        Args:
            ethical_storage: Instance of EthicalStorage
        """
        self.storage = ethical_storage
        self.output_dir = Path('reports/ethical_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_analysis_report(self, time_window_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive analysis of ethical decisions."""
        try:
            decisions = self.storage.get_recent_decisions(limit=10000)  # Get a large sample
            
            if not decisions:
                return {"error": "No decisions found for analysis"}

            # Convert to DataFrame for analysis
            df = pd.DataFrame([asdict(d) for d in decisions])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by time window
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            df = df[df['timestamp'] > cutoff_date]

            report = {
                "generated_at": datetime.now().isoformat(),
                "time_window_days": time_window_days,
                "total_decisions": len(df),
                "summary_metrics": self._calculate_summary_metrics(df),
                "trend_analysis": self._analyze_trends(df),
                "guideline_violations": self._analyze_guideline_violations(df),
                "action_type_analysis": self._analyze_action_types(df),
                "confidence_analysis": self._analyze_confidence_levels(df)
            }

            # Generate visualizations
            self._generate_visualizations(df)
            
            # Save report
            report_path = self.output_dir / f"ethical_analysis_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            return report

        except Exception as e:
            logger.error(f"Error generating analysis report: {str(e)}")
            return {"error": str(e)}

    def _calculate_summary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary metrics from decisions."""
        return {
            "approval_rate": (df['is_approved'].mean() * 100).round(2),
            "average_confidence": df['confidence'].mean().round(3),
            "total_violations": len(df[~df['is_approved']]),
            "unique_action_types": df['action_type'].nunique(),
            "decisions_per_day": (len(df) / ((df['timestamp'].max() - df['timestamp'].min()).days + 1)).round(2)
        }

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in ethical decisions."""
        daily_stats = df.set_index('timestamp').resample('D').agg({
            'is_approved': 'mean',
            'confidence': 'mean'
        }).fillna(method='ffill')

        return {
            "daily_approval_rates": daily_stats['is_approved'].to_dict(),
            "daily_confidence_levels": daily_stats['confidence'].to_dict(),
            "trend_direction": {
                "approval_rate": "increasing" if daily_stats['is_approved'].diff().mean() > 0 else "decreasing",
                "confidence": "increasing" if daily_stats['confidence'].diff().mean() > 0 else "decreasing"
            }
        }

    def _analyze_guideline_violations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze patterns in guideline violations."""
        violations = defaultdict(int)
        
        for _, row in df[~df['is_approved']].iterrows():
            for guideline in row['violated_guidelines']:
                violations[guideline] += 1

        return {
            "most_common_violations": sorted(violations.items(), key=lambda x: x[1], reverse=True),
            "violation_count_by_type": dict(violations)
        }

    def _analyze_action_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze different action types and their approval rates."""
        action_analysis = df.groupby('action_type').agg({
            'is_approved': ['count', 'mean'],
            'confidence': 'mean'
        }).round(3)

        return {
            "action_type_counts": action_analysis['is_approved']['count'].to_dict(),
            "action_type_approval_rates": action_analysis['is_approved']['mean'].to_dict(),
            "action_type_confidence": action_analysis['confidence']['mean'].to_dict()
        }

    def _analyze_confidence_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confidence levels in decisions."""
        confidence_bins = pd.cut(df['confidence'], bins=5)
        confidence_analysis = df.groupby(confidence_bins)['is_approved'].agg(['count', 'mean'])

        return {
            "confidence_distribution": confidence_analysis['count'].to_dict(),
            "approval_rate_by_confidence": confidence_analysis['mean'].to_dict(),
            "correlation_with_approval": df['confidence'].corr(df['is_approved'])
        }

    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization plots for the analysis."""
        try:
            # Set style
            plt.style.use('seaborn')
            
            # 1. Time series plot of approval rates
            plt.figure(figsize=(12, 6))
            daily_approvals = df.set_index('timestamp')['is_approved'].resample('D').mean()
            daily_approvals.plot(title='Daily Approval Rate Trend')
            plt.savefig(self.output_dir / 'approval_trend.png')
            plt.close()

            # 2. Action type analysis
            plt.figure(figsize=(10, 6))
            action_type_stats = df.groupby('action_type')['is_approved'].mean().sort_values()
            action_type_stats.plot(kind='barh', title='Approval Rate by Action Type')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'action_type_analysis.png')
            plt.close()

            # 3. Confidence distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='confidence', hue='is_approved', multiple="stack")
            plt.title('Confidence Distribution by Decision Outcome')
            plt.savefig(self.output_dir / 'confidence_distribution.png')
            plt.close()

            # 4. Violation types heatmap
            if len(df) > 0:
                violation_matrix = self._create_violation_matrix(df)
                plt.figure(figsize=(12, 8))
                sns.heatmap(violation_matrix, annot=True, cmap='YlOrRd')
                plt.title('Guideline Violation Patterns')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'violation_patterns.png')
                plt.close()

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")

    def _create_violation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a matrix of violation patterns."""
        violations = defaultdict(lambda: defaultdict(int))
        
        for _, row in df[~df['is_approved']].iterrows():
            for guideline in row['violated_guidelines']:
                violations[row['action_type']][guideline] += 1

        return pd.DataFrame(violations).fillna(0)

    def calculate_risk_metrics(self, time_window_days: int = 7) -> Dict[str, Any]:
        """Calculate risk metrics for ethical decision making."""
        try:
            recent_decisions = self.storage.get_recent_decisions(limit=1000)
            df = pd.DataFrame([asdict(d) for d in recent_decisions])
            
            if len(df) == 0:
                return {"error": "No decisions found for risk analysis"}

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            df = df[df['timestamp'] > cutoff_date]

            return {
                "risk_score": self._calculate_risk_score(df),
                "high_risk_actions": self._identify_high_risk_actions(df),
                "risk_trends": self._analyze_risk_trends(df),
                "recommendation": self._generate_risk_recommendations(df)
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {"error": str(e)}

    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """Calculate overall risk score."""
        if len(df) == 0:
            return 0.0

        weights = {
            'rejection_rate': 0.4,
            'low_confidence': 0.3,
            'violation_severity': 0.3
        }

        rejection_rate = 1 - df['is_approved'].mean()
        low_confidence = (df['confidence'] < 0.6).mean()
        violation_severity = len(df['violated_guidelines'].explode().unique()) / 10  # Normalized by max expected violations

        risk_score = (
            rejection_rate * weights['rejection_rate'] +
            low_confidence * weights['low_confidence'] +
            violation_severity * weights['violation_severity']
        )

        return round(risk_score * 100, 2)

    def _identify_high_risk_actions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify high-risk action patterns."""
        high_risk_actions = []
        
        for action_type in df['action_type'].unique():
            action_df = df[df['action_type'] == action_type]
            
            if len(action_df) < 5:  # Skip actions with too few samples
                continue

            risk_factors = []
            rejection_rate = 1 - action_df['is_approved'].mean()
            avg_confidence = action_df['confidence'].mean()
            
            if rejection_rate > 0.3:
                risk_factors.append("High rejection rate")
            if avg_confidence < 0.7:
                risk_factors.append("Low confidence")
            if len(action_df['violated_guidelines'].explode().unique()) > 3:
                risk_factors.append("Multiple guideline violations")

            if risk_factors:
                high_risk_actions.append({
                    "action_type": action_type,
                    "risk_factors": risk_factors,
                    "rejection_rate": round(rejection_rate * 100, 2),
                    "avg_confidence": round(avg_confidence, 3)
                })

        return sorted(high_risk_actions, key=lambda x: x['rejection_rate'], reverse=True)

    def _analyze_risk_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in risk factors."""
        daily_stats = df.set_index('timestamp').resample('D').agg({
            'is_approved': 'mean',
            'confidence': 'mean'
        }).fillna(method='ffill')

        return {
            "trend": {
                "risk_increasing": daily_stats['is_approved'].diff().mean() < 0,
                "confidence_trend": "decreasing" if daily_stats['confidence'].diff().mean() < 0 else "increasing"
            },
            "daily_risk_scores": daily_stats.apply(lambda x: self._calculate_risk_score(pd.DataFrame([x]))).to_dict()
        }

    def _generate_risk_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on risk analysis."""
        recommendations = []
        
        rejection_rate = 1 - df['is_approved'].mean()
        avg_confidence = df['confidence'].mean()
        violation_counts = df['violated_guidelines'].explode().value_counts()

        if rejection_rate > 0.3:
            recommendations.append(
                "High rejection rate detected. Review and update ethical guidelines "
                "or implement additional pre-checks for common violation patterns."
            )

        if avg_confidence < 0.7:
            recommendations.append(
                "Low average confidence in decisions. Consider implementing additional "
                "validation steps or gathering more context for decision-making."
            )

        if len(violation_counts) > 0:
            most_common_violation = violation_counts.index[0]
            recommendations.append(
                f"Most frequent guideline violation: {most_common_violation}. "
                "Consider focused training or guideline refinement in this area."
            )

        return recommendations

    def export_analysis(self, format: str = 'json') -> Optional[str]:
        """Export analysis results to file."""
        try:
            analysis = self.generate_analysis_report()
            risk_metrics = self.calculate_risk_metrics()
            
            export_data = {
                "analysis_report": analysis,
                "risk_metrics": risk_metrics,
                "export_timestamp": datetime.now().isoformat()
            }

            if format == 'json':
                export_path = self.output_dir / f"ethical_analysis_export_{datetime.now().strftime('%Y%m%d')}.json"
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            return str(export_path)

        except Exception as e:
            logger.error(f"Error exporting analysis: {str(e)}")
            return None
