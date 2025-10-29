"""
QuARA - Quantitative Academic Research Agent
A multi-agent system for autonomous scientific discovery and analysis
"""

__version__ = "0.1.0"
__author__ = "QuARA Team"
__description__ = "Multi-agent framework for quantitative academic research"

from .core import QuARASystem
from .agents import *
from .mcp import MCPHub
from .memory import ZettelkastenMemory
from .utils import *

__all__ = [
    "QuARASystem",
    "MCPHub", 
    "ZettelkastenMemory"
]