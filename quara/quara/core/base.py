"""
Base classes and interfaces for the QuARA multi-agent system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from enum import Enum
import uuid
import asyncio
from datetime import datetime


class TaskStatus(Enum):
    """Status of a task in the system"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(Enum):
    """Roles for different agents in the system"""
    ORCHESTRATOR = "orchestrator"
    THEORIST = "theorist"
    LIBRARIAN = "librarian"
    METHODOLOGIST = "methodologist"
    ANALYST = "analyst"
    SCRIBE = "scribe"


class Task(BaseModel):
    """Standard task format for MCP communication"""
    task_id: str = None
    originator: str
    target_agent: str
    goal: str
    context: Dict[str, Any] = {}
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    updated_at: datetime = None
    
    def __init__(self, **data):
        if data.get('task_id') is None:
            data['task_id'] = f"task_{uuid.uuid4().hex[:8]}"
        if data.get('created_at') is None:
            data['created_at'] = datetime.now()
        data['updated_at'] = datetime.now()
        super().__init__(**data)


class ToolRequest(BaseModel):
    """Standard tool request format"""
    request_id: str = None
    task_id: str
    originator: str
    tool_name: str
    parameters: Dict[str, Any]
    created_at: datetime = None
    
    def __init__(self, **data):
        if data.get('request_id') is None:
            data['request_id'] = f"req_{uuid.uuid4().hex[:8]}"
        if data.get('created_at') is None:
            data['created_at'] = datetime.now()
        super().__init__(**data)


class TaskResult(BaseModel):
    """Standard result format for completed tasks"""
    task_id: str
    status: TaskStatus
    result: Dict[str, Any] = {}
    error: Optional[str] = None
    completed_at: datetime = None
    
    def __init__(self, **data):
        if data.get('completed_at') is None:
            data['completed_at'] = datetime.now()
        super().__init__(**data)


class StandardizedAgentInterface(ABC):
    """
    Standardized Agent Interface (SAI) that all agents must implement
    """
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.active_tasks: Dict[str, Task] = {}
    
    @abstractmethod
    async def receive_task(self, task: Task) -> None:
        """Receive a new task from the MCP"""
        pass
    
    @abstractmethod
    async def report_status(self, task_id: str, status: TaskStatus) -> None:
        """Report task status to the MCP"""
        pass
    
    @abstractmethod
    async def return_result(self, task_id: str, result: TaskResult) -> None:
        """Return task result to the MCP"""
        pass
    
    @abstractmethod
    async def request_tool_use(self, tool_request: ToolRequest) -> Dict[str, Any]:
        """Request tool use through the MCP"""
        pass


class StandardizedToolInterface(ABC):
    """
    Standardized Tool Interface (STI) that all tools must implement
    """
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return OpenAPI-like schema for this tool"""
        pass


class BaseReActAgent(StandardizedAgentInterface):
    """
    Base ReAct (Reasoning and Acting) agent implementation
    All specialist agents inherit from this class
    """
    
    def __init__(self, agent_id: str, role: AgentRole, llm_client=None):
        super().__init__(agent_id, role)
        self.llm_client = llm_client
        self.tools = {}
        self.memory = []
        self.max_iterations = 10
    
    async def think(self, observation: str, context: Dict[str, Any]) -> str:
        """ReAct thinking step - analyze situation and plan next action"""
        prompt = f"""
        You are a {self.role.value} agent in a multi-agent research system.
        
        Current situation: {observation}
        Context: {context}
        Previous memory: {self.memory[-5:] if self.memory else 'None'}
        
        Think step by step:
        1. What is the current situation?
        2. What is my goal?
        3. What should I do next?
        
        Provide your reasoning:
        """
        
        if self.llm_client:
            response = await self._query_llm(prompt)
            return response
        else:
            return f"Analyzing situation for {self.role.value}"
    
    async def act(self, thought: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct acting step - take action based on reasoning"""
        prompt = f"""
        Based on your thought: {thought}
        Available tools: {list(self.tools.keys())}
        
        Choose an action:
        1. Use a tool (specify tool name and parameters)
        2. Complete the task (if goal is achieved)
        3. Request more information
        
        Format your response as JSON:
        {{
            "action_type": "use_tool|complete|request_info",
            "tool_name": "tool_name_if_applicable",
            "parameters": {{}},
            "reasoning": "why you chose this action"
        }}
        """
        
        if self.llm_client:
            response = await self._query_llm(prompt)
            try:
                import json
                return json.loads(response)
            except:
                return {"action_type": "complete", "reasoning": "Could not parse action"}
        else:
            return {
                "action_type": "complete",
                "reasoning": f"Mock action for {self.role.value}"
            }
    
    async def react_loop(self, task: Task) -> TaskResult:
        """Main ReAct loop for task execution"""
        observation = f"Received task: {task.goal}"
        context = task.context
        
        for iteration in range(self.max_iterations):
            # Think
            thought = await self.think(observation, context)
            self.memory.append(f"Thought {iteration + 1}: {thought}")
            
            # Act
            action = await self.act(thought, context)
            self.memory.append(f"Action {iteration + 1}: {action}")
            
            # Execute action
            if action["action_type"] == "complete":
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result={"reasoning": action.get("reasoning", "Task completed")}
                )
            elif action["action_type"] == "use_tool":
                try:
                    tool_result = await self._use_tool(
                        action["tool_name"], 
                        action.get("parameters", {}),
                        task.task_id
                    )
                    observation = f"Tool result: {tool_result}"
                    context["last_tool_result"] = tool_result
                except Exception as e:
                    observation = f"Tool error: {str(e)}"
            else:
                observation = "Continuing analysis..."
        
        # Max iterations reached
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Maximum iterations reached without completion"
        )
    
    async def _use_tool(self, tool_name: str, parameters: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Use a tool through the MCP"""
        tool_request = ToolRequest(
            task_id=task_id,
            originator=self.agent_id,
            tool_name=tool_name,
            parameters=parameters
        )
        return await self.request_tool_use(tool_request)
    
    async def _query_llm(self, prompt: str) -> str:
        """Query the language model"""
        if self.llm_client:
            # This would be implemented with actual LLM client
            return f"LLM response to: {prompt[:100]}..."
        return "Mock LLM response"


__all__ = [
    "TaskStatus",
    "AgentRole", 
    "Task",
    "ToolRequest",
    "TaskResult",
    "StandardizedAgentInterface",
    "StandardizedToolInterface",
    "BaseReActAgent"
]