# src/core/__init__.py

from .autonomous_learning import AutonomousLearning
from .natural_language_processing import NaturalLanguageProcessing
from .enhanced_knowledge_base import EnhancedKnowledgeBase
from .knowledge_base import KnowledgeBase
from .learning_system import LearningSystem
from .reasoning_engine import ReasoningEngine
from .knowledge_extractor import KnowledgeExtractor
from .task_specific_modules import TaskSpecificModules

__all__ = [
    'NaturalLanguageProcessing',
    'KnowledgeBase',
    'ReasoningEngine',
    'KnowledgeExtractor',
    'TaskSpecificModules'
]
