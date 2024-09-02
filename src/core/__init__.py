# src/core/__init__.py

from .natural_language_processing import NaturalLanguageProcessing
from .knowledge_base import KnowledgeBase
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
