"""
Agents module for QuARA multi-agent system
"""

from .orchestrator import OrchestratorAgent
from .theorist import TheoristAgent
from .librarian import LibrarianAgent
from .methodologist import MethodologistAgent
from .analyst import AnalystAgent, DataIntegrityModule, StatisticalCodeModule, CausalInferenceModule, VisualizationModule
from .scribe import ScribeAgent, IntroWriterAgent, MethodsWriterAgent, ResultsWriterAgent, DiscussionWriterAgent

__all__ = [
    "OrchestratorAgent",
    "TheoristAgent", 
    "LibrarianAgent",
    "MethodologistAgent",
    "AnalystAgent",
    "ScribeAgent",
    "DataIntegrityModule",
    "StatisticalCodeModule", 
    "CausalInferenceModule",
    "VisualizationModule",
    "IntroWriterAgent",
    "MethodsWriterAgent",
    "ResultsWriterAgent", 
    "DiscussionWriterAgent"
]