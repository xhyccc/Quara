"""
Master Control Protocol (MCP) Hub - Central communication system for QuARA agents
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
from collections import defaultdict

from ..core.base import (
    Task, TaskResult, ToolRequest, TaskStatus, 
    StandardizedAgentInterface, StandardizedToolInterface,
    AgentRole
)


class MCPHub:
    """
    Master Control Protocol Hub - Central message-passing bus for multi-agent communication
    
    This implements a publish/subscribe model where:
    - Orchestrator publishes high-level tasks
    - Specialist agents subscribe to relevant tasks
    - All tool usage is brokered through the hub for security
    """
    
    def __init__(self):
        self.agents: Dict[str, StandardizedAgentInterface] = {}
        self.tools: Dict[str, StandardizedToolInterface] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.tool_requests: asyncio.Queue = asyncio.Queue()
        
        # Security policies for agent-tool access
        self.security_policies: Dict[str, List[str]] = {
            "orchestrator": ["*"],  # Full access
            "theorist": ["web_search", "academic_apis", "llm"],
            "librarian": ["web_search", "academic_apis", "kaggle_api", "uci_api"],
            "methodologist": ["academic_apis", "statistical_libs", "llm"],
            "analyst": ["statistical_libs", "visualization", "sandbox", "data_cleaning"],
            "scribe": ["llm", "document_tools"]
        }
        
        # Message routing tables
        self.task_subscribers: Dict[str, List[str]] = defaultdict(list)
        self.result_callbacks: Dict[str, Callable] = {}
        
        # Logging and observability
        self.message_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("MCPHub")
        
        # Hub state
        self.running = False
        self._hub_task = None
    
    async def start(self):
        """Start the MCP hub"""
        self.running = True
        self._hub_task = asyncio.create_task(self._hub_loop())
        self.logger.info("MCP Hub started")
    
    async def stop(self):
        """Stop the MCP hub"""
        self.running = False
        if self._hub_task:
            self._hub_task.cancel()
            try:
                await self._hub_task
            except asyncio.CancelledError:
                pass
        self.logger.info("MCP Hub stopped")
    
    def register_agent(self, agent: StandardizedAgentInterface) -> None:
        """Register an agent with the hub"""
        self.agents[agent.agent_id] = agent
        
        # Subscribe agent to relevant tasks
        role_name = agent.role.value
        self.task_subscribers[role_name].append(agent.agent_id)
        
        self.logger.info(f"Registered agent: {agent.agent_id} (role: {role_name})")
    
    def register_tool(self, tool_name: str, tool: StandardizedToolInterface) -> None:
        """Register a tool with the hub"""
        self.tools[tool_name] = tool
        self.logger.info(f"Registered tool: {tool_name}")
    
    async def publish_task(self, task: Task) -> str:
        """Publish a task to the hub"""
        task.status = TaskStatus.PENDING
        await self.task_queue.put(task)
        
        # Log the task
        self._log_message({
            "type": "task_published",
            "task_id": task.task_id,
            "originator": task.originator,
            "target_agent": task.target_agent,
            "goal": task.goal,
            "timestamp": datetime.now().isoformat()
        })
        
        return task.task_id
    
    async def publish_result(self, result: TaskResult) -> None:
        """Publish a task result"""
        await self.result_queue.put(result)
        
        # Log the result
        self._log_message({
            "type": "result_published",
            "task_id": result.task_id,
            "status": result.status.value,
            "timestamp": datetime.now().isoformat()
        })
    
    async def request_tool_use(self, tool_request: ToolRequest) -> Dict[str, Any]:
        """Request tool usage through the hub with security checks"""
        
        # Security check
        if not self._check_tool_permission(tool_request.originator, tool_request.tool_name):
            error_msg = f"Agent {tool_request.originator} not authorized to use tool {tool_request.tool_name}"
            self.logger.warning(error_msg)
            return {"error": error_msg, "authorized": False}
        
        # Check if tool exists
        if tool_request.tool_name not in self.tools:
            error_msg = f"Tool {tool_request.tool_name} not found"
            self.logger.error(error_msg)
            return {"error": error_msg, "tool_found": False}
        
        try:
            # Execute the tool
            tool = self.tools[tool_request.tool_name]
            result = await tool.execute(tool_request.parameters)
            
            # Log successful tool use
            self._log_message({
                "type": "tool_executed",
                "request_id": tool_request.request_id,
                "tool_name": tool_request.tool_name,
                "originator": tool_request.originator,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
            
            return {"success": True, "result": result}
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Log failed tool use
            self._log_message({
                "type": "tool_executed",
                "request_id": tool_request.request_id,
                "tool_name": tool_request.tool_name,
                "originator": tool_request.originator,
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            return {"success": False, "error": error_msg}
    
    async def _hub_loop(self):
        """Main hub processing loop"""
        while self.running:
            try:
                # Process tasks
                if not self.task_queue.empty():
                    task = await self.task_queue.get()
                    await self._route_task(task)
                
                # Process results
                if not self.result_queue.empty():
                    result = await self.result_queue.get()
                    await self._route_result(result)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in hub loop: {str(e)}")
    
    async def _route_task(self, task: Task) -> None:
        """Route a task to the appropriate agent"""
        target_agent_id = None
        
        # Find target agent
        if task.target_agent in self.agents:
            target_agent_id = task.target_agent
        else:
            # Find by role
            for agent_id in self.task_subscribers.get(task.target_agent, []):
                if agent_id in self.agents:
                    target_agent_id = agent_id
                    break
        
        if target_agent_id:
            agent = self.agents[target_agent_id]
            try:
                await agent.receive_task(task)
                self.logger.info(f"Routed task {task.task_id} to agent {target_agent_id}")
            except Exception as e:
                self.logger.error(f"Failed to route task {task.task_id}: {str(e)}")
        else:
            self.logger.error(f"No agent found for task {task.task_id} targeting {task.target_agent}")
    
    async def _route_result(self, result: TaskResult) -> None:
        """Route a result back to the originator"""
        # This would typically route back to the Orchestrator
        # For now, we just log it
        self.logger.info(f"Result received for task {result.task_id}: {result.status.value}")
    
    def _check_tool_permission(self, agent_id: str, tool_name: str) -> bool:
        """Check if an agent has permission to use a tool"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        role_name = agent.role.value
        
        if role_name not in self.security_policies:
            return False
        
        allowed_tools = self.security_policies[role_name]
        
        # Check for wildcard access
        if "*" in allowed_tools:
            return True
        
        # Check for specific tool access
        return tool_name in allowed_tools
    
    def _log_message(self, message: Dict[str, Any]) -> None:
        """Log a message for observability"""
        self.message_log.append(message)
        
        # Keep only last 10000 messages to prevent memory issues
        if len(self.message_log) > 10000:
            self.message_log = self.message_log[-5000:]
    
    def get_message_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the message log for observability"""
        if limit:
            return self.message_log[-limit:]
        return self.message_log.copy()
    
    def get_hub_status(self) -> Dict[str, Any]:
        """Get current hub status"""
        return {
            "running": self.running,
            "agents_count": len(self.agents),
            "tools_count": len(self.tools),
            "pending_tasks": self.task_queue.qsize(),
            "pending_results": self.result_queue.qsize(),
            "total_messages": len(self.message_log),
            "agents": {
                agent_id: agent.role.value 
                for agent_id, agent in self.agents.items()
            },
            "tools": list(self.tools.keys())
        }


class MockTool(StandardizedToolInterface):
    """Mock tool for testing"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tool": self.name,
            "parameters": parameters,
            "result": f"Mock result from {self.name}",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": f"Mock {self.name} tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query parameter"}
                }
            }
        }


__all__ = ["MCPHub", "MockTool"]