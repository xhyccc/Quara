"""
Main QuARA System - Complete multi-agent research framework
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os

from ..core.base import Task, TaskResult, TaskStatus, AgentRole
from ..mcp.hub import MCPHub, MockTool
from ..agents import (
    OrchestratorAgent, TheoristAgent, LibrarianAgent,
    MethodologistAgent, AnalystAgent, ScribeAgent
)
from ..memory.zettelkasten import ZettelkastenMemory
from ..utils.llm_client import LLMClient, LLMConfig, create_llm_client


class QuARASystem:
    """
    Complete QuARA (Quantitative Academic Research Agent) system
    
    This is the main entry point that orchestrates the entire multi-agent
    research workflow from initial query to final publication.
    """
    
    def __init__(self, 
                 llm_client=None,
                 llm_config: Optional[Dict[str, Any]] = None,
                 enable_memory: bool = True,
                 enable_real_tools: bool = False,
                 project_root: str = "./research_projects"):
        
        # Initialize LLM client
        if llm_client is None:
            if llm_config:
                # Create from config dict
                config = LLMConfig.from_dict(llm_config)
                self.llm_client = LLMClient(config)
            else:
                # Try to create from environment variables
                try:
                    self.llm_client = create_llm_client()
                except (ValueError, ImportError) as e:
                    logging.warning(f"Failed to initialize LLM client: {e}")
                    logging.warning("Running without LLM - agents will use mock responses")
                    self.llm_client = None
        else:
            self.llm_client = llm_client
        
        # Core system components
        self.mcp_hub = MCPHub()
        self.memory = ZettelkastenMemory() if enable_memory else None
        self.project_root = project_root
        
        # Ensure project directory exists
        os.makedirs(project_root, exist_ok=True)
        
        # Initialize all agents with LLM client
        self.agents = {
            "orchestrator": OrchestratorAgent("orchestrator", self.llm_client, self.mcp_hub),
            "theorist": TheoristAgent("theorist", self.llm_client, self.mcp_hub),
            "librarian": LibrarianAgent("librarian", self.llm_client, self.mcp_hub),
            "methodologist": MethodologistAgent("methodologist", self.llm_client, self.mcp_hub),
            "analyst": AnalystAgent("analyst", self.llm_client, self.mcp_hub),
            "scribe": ScribeAgent("scribe", self.llm_client, self.mcp_hub)
        }
        
        # Register agents with MCP hub
        for agent in self.agents.values():
            self.mcp_hub.register_agent(agent)
        
        # Register tools (mock tools for demo)
        if not enable_real_tools:
            self._setup_mock_tools()
        
        self.logger = logging.getLogger("QuARASystem")
        self.system_running = False
    
    def _setup_mock_tools(self):
        """Setup mock tools for demonstration"""
        mock_tools = [
            "web_search", "pubmed_api", "arxiv_api", "kaggle_api",
            "obi_ontology", "causal_classifier", "model_selector",
            "autodcworkflow", "e2b_sandbox", "dowhy_causal",
            "llm", "document_tools"
        ]
        
        for tool_name in mock_tools:
            self.mcp_hub.register_tool(tool_name, MockTool(tool_name))
    
    async def start_system(self):
        """Start the QuARA system"""
        if self.system_running:
            return
        
        await self.mcp_hub.start()
        self.system_running = True
        self.logger.info("QuARA system started successfully")
    
    async def stop_system(self):
        """Stop the QuARA system"""
        if not self.system_running:
            return
        
        await self.mcp_hub.stop()
        self.system_running = False
        self.logger.info("QuARA system stopped")
    
    async def conduct_research(self, research_request: str, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Conduct complete research workflow from request to final paper
        
        This is the main public interface for the QuARA system.
        """
        
        if not self.system_running:
            await self.start_system()
        
        # Create initial task for the orchestrator
        task = Task(
            originator="user",
            target_agent="orchestrator",
            goal=research_request,
            context=context or {}
        )
        
        try:
            # Execute research through orchestrator
            self.logger.info(f"Starting research: {research_request}")
            
            # Store initial request in memory if available
            if self.memory:
                await self.memory.store_knowledge(
                    content=research_request,
                    node_type="research_request",
                    metadata={"timestamp": datetime.now().isoformat(), "source": "user"}
                )
            
            # Get orchestrator and execute
            orchestrator = self.agents["orchestrator"]
            
            # Simulate the research workflow
            result = await orchestrator.execute_research_workflow(task)
            
            # Store final results in memory
            if self.memory and result.status == TaskStatus.COMPLETED:
                await self.memory.store_knowledge(
                    content=f"Research completed: {result.result}",
                    node_type="research_result",
                    project_id=result.result.get("project_id"),
                    metadata={"timestamp": datetime.now().isoformat(), "status": "completed"}
                )
            
            return {
                "success": result.status == TaskStatus.COMPLETED,
                "project_id": result.result.get("project_id") if result.status == TaskStatus.COMPLETED else None,
                "result": result.result if result.status == TaskStatus.COMPLETED else None,
                "error": result.error if result.status == TaskStatus.FAILED else None,
                "system_status": self.get_system_status()
            }
            
        except Exception as e:
            self.logger.error(f"Research workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "system_status": self.get_system_status()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "running": self.system_running,
            "mcp_hub_status": self.mcp_hub.get_hub_status(),
            "agents": {
                agent_id: {
                    "role": agent.role.value,
                    "active_tasks": len(agent.active_tasks)
                }
                for agent_id, agent in self.agents.items()
            },
            "memory_stats": self.memory.get_memory_statistics() if self.memory else None
        }
    
    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get status of a specific research project"""
        
        # Get project memory summary
        memory_summary = None
        if self.memory:
            memory_summary = await self.memory.get_project_memory_summary(project_id)
        
        # Check if project directory exists
        project_dir = os.path.join(self.project_root, project_id)
        project_exists = os.path.exists(project_dir)
        
        artifacts = []
        if project_exists:
            # List project artifacts
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), project_dir)
                    artifacts.append(rel_path)
        
        return {
            "project_id": project_id,
            "project_exists": project_exists,
            "project_directory": project_dir,
            "artifacts": artifacts,
            "memory_summary": memory_summary
        }
    
    async def query_system_memory(self, query: str, project_id: str = None) -> List[Dict[str, Any]]:
        """Query the system's long-term memory"""
        if not self.memory:
            return []
        
        nodes = await self.memory.query_memory(query, project_id=project_id)
        
        return [
            {
                "node_id": node.node_id,
                "content": node.content,
                "node_type": node.node_type,
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
                "tags": list(node.tags),
                "importance_score": node.importance_score
            }
            for node in nodes
        ]
    
    async def get_message_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get system message log for observability"""
        return self.mcp_hub.get_message_log(limit)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.system_running:
            asyncio.create_task(self.stop_system())
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_system()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_system()


# Convenience function for simple usage
async def conduct_research(research_request: str, 
                         context: Dict[str, Any] = None,
                         **system_kwargs) -> Dict[str, Any]:
    """
    Convenience function to conduct research with default settings
    
    Example:
        result = await conduct_research(
            "Analyze the effect of screen time on adolescent sleep quality"
        )
    """
    
    system = QuARASystem(**system_kwargs)
    
    try:
        return await system.conduct_research(research_request, context)
    finally:
        await system.stop_system()


__all__ = ["QuARASystem", "conduct_research"]