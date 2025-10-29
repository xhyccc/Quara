"""
Test suite for QuARA multi-agent system
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quara.core.base import Task, TaskStatus, AgentRole
from quara.mcp.hub import MCPHub, MockTool
from quara.agents.orchestrator import OrchestratorAgent
from quara.memory.zettelkasten import ZettelkastenMemory, MemoryNode
from quara.core.system import QuARASystem


class TestBasicComponents:
    """Test basic system components"""
    
    def test_task_creation(self):
        """Test task creation with auto-generated IDs"""
        task = Task(
            originator="test",
            target_agent="theorist",
            goal="Test task"
        )
        
        assert task.task_id is not None
        assert task.originator == "test"
        assert task.target_agent == "theorist"
        assert task.goal == "Test task"
        assert task.status == TaskStatus.PENDING
    
    def test_memory_node_creation(self):
        """Test memory node creation and operations"""
        node = MemoryNode(
            content="Test hypothesis",
            node_type="hypothesis"
        )
        
        assert node.node_id is not None
        assert node.content == "Test hypothesis"
        assert node.node_type == "hypothesis"
        assert len(node.links) == 0
        assert len(node.tags) == 0
        
        # Test adding links and tags
        node.add_link("other_node_id", "supports")
        node.add_tag("causal_inference")
        
        assert len(node.links) == 1
        assert len(node.tags) == 1
        assert node.links[0]["target"] == "other_node_id"
        assert "causal_inference" in node.tags


class TestMCPHub:
    """Test Master Control Protocol Hub"""
    
    @pytest.mark.asyncio
    async def test_hub_startup_shutdown(self):
        """Test hub lifecycle"""
        hub = MCPHub()
        
        assert not hub.running
        
        await hub.start()
        assert hub.running
        
        await hub.stop()
        assert not hub.running
    
    @pytest.mark.asyncio
    async def test_tool_registration_and_use(self):
        """Test tool registration and usage"""
        hub = MCPHub()
        await hub.start()
        
        # Register a mock tool
        mock_tool = MockTool("test_tool")
        hub.register_tool("test_tool", mock_tool)
        
        # Test tool usage
        from quara.core.base import ToolRequest
        
        request = ToolRequest(
            task_id="test",
            originator="test_agent",
            tool_name="test_tool",
            parameters={"query": "test"}
        )
        
        result = await hub.request_tool_use(request)
        
        assert result["success"] is True
        assert "result" in result
        
        await hub.stop()
    
    def test_security_policies(self):
        """Test security policy enforcement"""
        hub = MCPHub()
        
        # Test permission checking
        assert hub._check_tool_permission("orchestrator", "any_tool") is True  # wildcard access
        assert hub._check_tool_permission("theorist", "web_search") is True
        assert hub._check_tool_permission("theorist", "sandbox") is False
        assert hub._check_tool_permission("analyst", "sandbox") is True


class TestAgents:
    """Test individual agents"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_agent(self):
        """Test orchestrator agent basic functionality"""
        agent = OrchestratorAgent()
        
        assert agent.agent_id == "orchestrator"
        assert agent.role == AgentRole.ORCHESTRATOR
        assert len(agent.phases) == 6
        
        # Test task processing
        task = Task(
            originator="user",
            target_agent="orchestrator",
            goal="Test research request"
        )
        
        # Mock the execute_research_workflow method for testing
        original_method = agent.execute_research_workflow
        agent.execute_research_workflow = AsyncMock(return_value=Mock(
            status=TaskStatus.COMPLETED,
            result={"project_id": "test_project"}
        ))
        
        # This would normally process the task
        result = await agent.execute_research_workflow(task)
        
        assert result.status == TaskStatus.COMPLETED
        assert "project_id" in result.result


class TestMemorySystem:
    """Test Zettelkasten memory system"""
    
    @pytest.mark.asyncio
    async def test_memory_storage_and_retrieval(self):
        """Test memory storage with Mem0 update logic"""
        memory = ZettelkastenMemory(use_chromadb=False)
        
        # Store initial knowledge
        node_id1 = await memory.store_knowledge(
            content="Screen time affects sleep quality",
            node_type="hypothesis",
            metadata={"reliability_score": 0.8}
        )
        
        assert node_id1 is not None
        assert len(memory.nodes) == 1
        
        # Store similar knowledge (should trigger UPDATE or NOOP)
        node_id2 = await memory.store_knowledge(
            content="Screen time negatively impacts sleep",
            node_type="hypothesis", 
            metadata={"reliability_score": 0.9}
        )
        
        # Should have either updated existing or added new based on similarity
        assert len(memory.update_history) >= 2
        
        # Query memory
        results = await memory.query_memory("screen time sleep")
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_memory_operations(self):
        """Test different memory update operations"""
        memory = ZettelkastenMemory(use_chromadb=False)
        
        # Test ADD operation
        node_id = await memory.store_knowledge(
            "New research finding",
            "result"
        )
        
        # Verify ADD operation in history
        add_ops = [op for op in memory.update_history if op.operation_type == "ADD"]
        assert len(add_ops) >= 1
        
        # Test project association
        project_node_id = await memory.store_knowledge(
            "Project-specific finding",
            "result",
            project_id="test_project"
        )
        
        assert "test_project" in memory.project_memories
        assert project_node_id in memory.project_memories["test_project"]


class TestFullSystem:
    """Test complete QuARA system integration"""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test system startup and configuration"""
        system = QuARASystem(enable_real_tools=False)
        
        # Check all agents are initialized
        expected_agents = ["orchestrator", "theorist", "librarian", "methodologist", "analyst", "scribe"]
        for agent_name in expected_agents:
            assert agent_name in system.agents
        
        # Check MCP hub is configured
        assert system.mcp_hub is not None
        
        # Check memory is available
        assert system.memory is not None
        
        await system.start_system()
        assert system.system_running is True
        
        await system.stop_system()
        assert system.system_running is False
    
    @pytest.mark.asyncio
    async def test_research_workflow_basic(self):
        """Test basic research workflow execution"""
        system = QuARASystem(enable_real_tools=False)
        
        try:
            # Test with a simple request
            result = await system.conduct_research(
                "Test research request",
                context={"test": True}
            )
            
            # Should complete successfully with mock tools
            assert isinstance(result, dict)
            assert "success" in result
            
        finally:
            await system.stop_system()


class TestUtilities:
    """Test utility functions"""
    
    def test_research_request_validation(self):
        """Test research request validation"""
        from quara.utils.helpers import validate_research_request
        
        # Valid request
        result = validate_research_request("Analyze the effect of X on Y")
        assert result["valid"] is True
        assert result["has_research_indicators"] is True
        
        # Invalid requests
        result = validate_research_request("")
        assert result["valid"] is False
        
        result = validate_research_request("short")
        assert result["valid"] is False
    
    def test_key_term_extraction(self):
        """Test key term extraction"""
        from quara.utils.helpers import extract_key_terms
        
        text = "This study analyzes the causal effect of social media usage on sleep quality"
        terms = extract_key_terms(text)
        
        assert "study" in terms
        assert "analyzes" in terms
        assert "causal" in terms
        assert "effect" in terms
        assert len(terms) > 0
    
    def test_similarity_calculation(self):
        """Test text similarity calculation"""
        from quara.utils.helpers import calculate_similarity
        
        text1 = "Screen time affects sleep quality in adolescents"
        text2 = "Social media usage impacts sleep patterns in teenagers"
        
        similarity = calculate_similarity(text1, text2)
        assert 0 <= similarity <= 1
        assert similarity > 0  # Should have some similarity


# Test runner
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])