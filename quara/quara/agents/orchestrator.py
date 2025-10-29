"""
Orchestrator Agent - The "Principal Investigator" of the multi-agent system
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..core.base import (
    BaseReActAgent, AgentRole, Task, TaskResult, TaskStatus, ToolRequest
)


class OrchestratorAgent(BaseReActAgent):
    """
    The Orchestrator Agent manages the high-level research workflow.
    
    Core functions:
    - Project Scoping (Phase 0 iterative design)
    - Planning and decomposition of research goals
    - Delegation to specialist agents
    - Synthesis of results from multiple agents
    - Human-in-the-Loop checkpoint management
    """
    
    def __init__(self, agent_id: str = "orchestrator", llm_client=None, mcp_hub=None):
        super().__init__(agent_id, AgentRole.ORCHESTRATOR, llm_client)
        
        self.mcp_hub = mcp_hub
        self.active_projects: Dict[str, Dict[str, Any]] = {}
        self.project_counter = 0
        self.hitl_checkpoints = []
        
        # Research workflow phases
        self.phases = [
            "phase_0_design",
            "phase_1_theorist", 
            "phase_2_librarian",
            "phase_3_methodologist",
            "phase_4_analyst",
            "phase_5_scribe"
        ]
        
        self.logger = logging.getLogger(f"Orchestrator.{agent_id}")
    
    async def receive_task(self, task: Task) -> None:
        """Receive and process a new research request"""
        self.active_tasks[task.task_id] = task
        self.logger.info(f"Received task: {task.goal}")
        
        # Start the research workflow
        result = await self.execute_research_workflow(task)
        await self.return_result(task.task_id, result)
    
    async def report_status(self, task_id: str, status: TaskStatus) -> None:
        """Report task status to MCP"""
        if self.mcp_hub:
            # This would update the MCP hub with status
            pass
        self.logger.info(f"Task {task_id} status: {status.value}")
    
    async def return_result(self, task_id: str, result: TaskResult) -> None:
        """Return completed result to MCP"""
        if self.mcp_hub:
            await self.mcp_hub.publish_result(result)
        self.logger.info(f"Returned result for task {task_id}")
    
    async def request_tool_use(self, tool_request: ToolRequest) -> Dict[str, Any]:
        """Request tool use through MCP"""
        if self.mcp_hub:
            return await self.mcp_hub.request_tool_use(tool_request)
        return {"result": "Mock tool result", "success": True}
    
    async def execute_research_workflow(self, task: Task) -> TaskResult:
        """Execute the complete research workflow"""
        project_id = f"research_project_{self.project_counter}"
        self.project_counter += 1
        
        # Initialize project
        project = {
            "id": project_id,
            "original_request": task.goal,
            "context": task.context,
            "phase": "phase_0_design",
            "created_at": datetime.now(),
            "directory": f"./{project_id}/",
            "artifacts": {},
            "workflow_state": {}
        }
        
        self.active_projects[project_id] = project
        
        try:
            # Phase 0: Iterative Research Design & User Validation
            await self.execute_phase_0(project)
            
            # Wait for user "go ahead" (HITL checkpoint)
            if await self.hitl_checkpoint("final_design_approval", project):
                # Create project directory
                await self.create_project_directory(project)
                
                # Execute remaining phases
                for phase in self.phases[1:]:  # Skip phase_0 as it's already done
                    project["phase"] = phase
                    await self.execute_phase(phase, project)
                    
                    # Check for HITL checkpoints
                    checkpoint_name = self.get_checkpoint_for_phase(phase)
                    if checkpoint_name:
                        if not await self.hitl_checkpoint(checkpoint_name, project):
                            return TaskResult(
                                task_id=task.task_id,
                                status=TaskStatus.FAILED,
                                error=f"User rejected at checkpoint: {checkpoint_name}"
                            )
                
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result={
                        "project_id": project_id,
                        "directory": project["directory"],
                        "artifacts": project["artifacts"],
                        "message": "Research workflow completed successfully"
                    }
                )
            else:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.CANCELLED,
                    error="User did not approve final design"
                )
        
        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    async def execute_phase_0(self, project: Dict[str, Any]) -> None:
        """Phase 0: Iterative Research Design & User Validation"""
        self.logger.info("Starting Phase 0: Iterative Research Design")
        
        original_request = project["original_request"]
        iterations = 0
        max_iterations = 3
        
        while iterations < max_iterations:
            self.logger.info(f"Design iteration {iterations + 1}")
            
            # Get initial search results from Librarian
            librarian_task = Task(
                originator=self.agent_id,
                target_agent="librarian",
                goal=f"Perform initial literature and data search for: {original_request}",
                context={"iteration": iterations + 1, "type": "initial_search"}
            )
            
            # This would delegate to Librarian agent through MCP
            search_results = await self.delegate_task(librarian_task)
            
            # Generate refined research proposal
            proposal = await self.generate_research_proposal(original_request, search_results, iterations)
            
            # Store proposal for user review
            project["workflow_state"][f"proposal_iteration_{iterations + 1}"] = proposal
            
            # In a real implementation, this would present to user for feedback
            # For now, we simulate user feedback
            user_feedback = await self.simulate_user_feedback(proposal, iterations)
            
            if user_feedback.get("approved", False):
                project["final_proposal"] = proposal
                break
            else:
                # Incorporate feedback for next iteration
                original_request = user_feedback.get("refined_request", original_request)
                iterations += 1
        
        if "final_proposal" not in project:
            raise Exception("Max design iterations reached without approval")
    
    async def execute_phase(self, phase: str, project: Dict[str, Any]) -> None:
        """Execute a specific phase of the research workflow"""
        self.logger.info(f"Executing {phase}")
        
        if phase == "phase_1_theorist":
            await self.execute_theorist_phase(project)
        elif phase == "phase_2_librarian":
            await self.execute_librarian_phase(project)
        elif phase == "phase_3_methodologist":
            await self.execute_methodologist_phase(project)
        elif phase == "phase_4_analyst":
            await self.execute_analyst_phase(project)
        elif phase == "phase_5_scribe":
            await self.execute_scribe_phase(project)
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    async def execute_theorist_phase(self, project: Dict[str, Any]) -> None:
        """Phase 1: Theorist - Problem Definition"""
        task = Task(
            originator=self.agent_id,
            target_agent="theorist",
            goal="Generate testable hypothesis from validated research topic",
            context={
                "research_topic": project["final_proposal"]["topic"],
                "literature_gaps": project["final_proposal"].get("gaps", []),
                "project_id": project["id"]
            }
        )
        
        result = await self.delegate_task(task)
        project["artifacts"]["hypothesis"] = result
        project["workflow_state"]["theorist_complete"] = True
    
    async def execute_librarian_phase(self, project: Dict[str, Any]) -> None:
        """Phase 2: Librarian - Data Collection & Domain Knowledge"""
        task = Task(
            originator=self.agent_id,
            target_agent="librarian",
            goal="Collect comprehensive data and domain knowledge",
            context={
                "hypothesis": project["artifacts"]["hypothesis"],
                "project_id": project["id"]
            }
        )
        
        result = await self.delegate_task(task)
        project["artifacts"]["data_sources"] = result
        project["workflow_state"]["librarian_complete"] = True
    
    async def execute_methodologist_phase(self, project: Dict[str, Any]) -> None:
        """Phase 3: Methodologist - Experimental Design"""
        task = Task(
            originator=self.agent_id,
            target_agent="methodologist",
            goal="Design rigorous experimental methodology",
            context={
                "hypothesis": project["artifacts"]["hypothesis"],
                "data_sources": project["artifacts"]["data_sources"],
                "project_id": project["id"]
            }
        )
        
        result = await self.delegate_task(task)
        project["artifacts"]["methodology"] = result
        project["workflow_state"]["methodologist_complete"] = True
    
    async def execute_analyst_phase(self, project: Dict[str, Any]) -> None:
        """Phase 4: Analyst - Quantitative Execution"""
        task = Task(
            originator=self.agent_id,
            target_agent="analyst",
            goal="Execute quantitative analysis with baseline/benchmark comparison",
            context={
                "methodology": project["artifacts"]["methodology"],
                "data_sources": project["artifacts"]["data_sources"],
                "project_id": project["id"]
            }
        )
        
        result = await self.delegate_task(task)
        project["artifacts"]["analysis_results"] = result
        project["workflow_state"]["analyst_complete"] = True
    
    async def execute_scribe_phase(self, project: Dict[str, Any]) -> None:
        """Phase 5: Scribe - Academic Writing"""
        task = Task(
            originator=self.agent_id,
            target_agent="scribe",
            goal="Generate complete academic manuscript",
            context={
                "hypothesis": project["artifacts"]["hypothesis"],
                "methodology": project["artifacts"]["methodology"],
                "results": project["artifacts"]["analysis_results"],
                "project_id": project["id"]
            }
        )
        
        result = await self.delegate_task(task)
        project["artifacts"]["manuscript"] = result
        project["workflow_state"]["scribe_complete"] = True
    
    async def delegate_task(self, task: Task) -> Dict[str, Any]:
        """Delegate a task to a specialist agent"""
        if self.mcp_hub:
            # Publish task to MCP hub
            task_id = await self.mcp_hub.publish_task(task)
            
            # Wait for result (in real implementation, this would be more sophisticated)
            # For now, return mock result based on task
            return await self.simulate_agent_response(task)
        else:
            return await self.simulate_agent_response(task)
    
    async def simulate_agent_response(self, task: Task) -> Dict[str, Any]:
        """Simulate specialist agent response (for testing)"""
        agent_type = task.target_agent
        
        if agent_type == "librarian":
            return {
                "type": "librarian_response",
                "literature": ["Paper 1", "Paper 2", "Paper 3"],
                "datasets": ["Dataset A", "Dataset B"],
                "domain_knowledge": "Key concepts and relationships"
            }
        elif agent_type == "theorist":
            return {
                "type": "theorist_response", 
                "hypothesis": "Screen time negatively affects sleep quality, moderated by anxiety",
                "variables": {"independent": "screen_time", "dependent": "sleep_quality", "moderator": "anxiety"}
            }
        elif agent_type == "methodologist":
            return {
                "type": "methodologist_response",
                "task_type": "causal_inference",
                "proposed_model": "Statsmodels OLS with interaction term",
                "evaluation_plan": "Compare against baseline and benchmark models"
            }
        elif agent_type == "analyst":
            return {
                "type": "analyst_response",
                "results": "Analysis complete with significant findings",
                "figures": ["figure_1.png", "figure_2.png"],
                "statistical_outputs": "Model summary tables"
            }
        elif agent_type == "scribe":
            return {
                "type": "scribe_response",
                "manuscript": "Complete academic paper with all sections",
                "format": "LaTeX"
            }
        
        return {"type": "unknown_agent_response"}
    
    async def generate_research_proposal(self, request: str, search_results: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Generate a refined research proposal"""
        return {
            "iteration": iteration + 1,
            "topic": f"Refined research topic based on: {request}",
            "specific_question": "A focused, testable research question",
            "potential_datasets": search_results.get("datasets", []),
            "gaps": ["Research gap 1", "Research gap 2"],
            "methodology_outline": "High-level approach"
        }
    
    async def simulate_user_feedback(self, proposal: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Simulate user feedback (in real implementation, this would be HITL)"""
        if iteration < 2:  # Simulate a few iterations
            return {
                "approved": False,
                "feedback": "Need more focus on specific aspect",
                "refined_request": proposal["topic"] + " with focus on specific demographics"
            }
        else:
            return {"approved": True}
    
    async def hitl_checkpoint(self, checkpoint_name: str, project: Dict[str, Any]) -> bool:
        """Human-in-the-Loop validation checkpoint"""
        self.logger.info(f"HITL Checkpoint: {checkpoint_name}")
        
        # In real implementation, this would:
        # 1. Pause the workflow
        # 2. Present information to human expert
        # 3. Wait for approval/rejection/modifications
        # 4. Resume based on response
        
        # For now, simulate approval
        checkpoint_data = {
            "checkpoint": checkpoint_name,
            "project_id": project["id"],
            "timestamp": datetime.now(),
            "data": project.get("artifacts", {}),
            "approved": True  # Simulate approval
        }
        
        self.hitl_checkpoints.append(checkpoint_data)
        return True  # Simulate user approval
    
    def get_checkpoint_for_phase(self, phase: str) -> Optional[str]:
        """Get the HITL checkpoint name for a phase"""
        checkpoint_map = {
            "phase_2_librarian": "data_source_validation",
            "phase_3_methodologist": "methodological_validation", 
            "phase_4_analyst": "results_validation",
            "phase_5_scribe": "final_manuscript_approval"
        }
        return checkpoint_map.get(phase)
    
    async def create_project_directory(self, project: Dict[str, Any]) -> None:
        """Create dedicated project directory"""
        # In real implementation, this would create actual directories
        self.logger.info(f"Creating project directory: {project['directory']}")
        project["directory_created"] = True