"""
Methodologist Agent - Experimental Design and Statistical Planning
"""

import asyncio
from typing import Dict, Any, List, Optional
import logging

from ..core.base import (
    BaseReActAgent, AgentRole, Task, TaskResult, TaskStatus, ToolRequest
)


class MethodologistAgent(BaseReActAgent):
    """
    The Methodologist Agent handles experimental design and statistical planning.
    
    Core functions:
    - Hypothesis formalization using ontologies (OBI)
    - Critical juncture: Causal vs. Predictive classification
    - Proposed model selection
    - Rigorous evaluation and comparison framework design
    """
    
    def __init__(self, agent_id: str = "methodologist", llm_client=None, mcp_hub=None):
        super().__init__(agent_id, AgentRole.METHODOLOGIST, llm_client)
        
        self.mcp_hub = mcp_hub
        self.methodology_frameworks = {}
        self.evaluation_templates = {}
        
        # Specialized tools for Methodologist
        self.tools = {
            "obi_ontology": "Ontology for Biomedical Investigations formalization",
            "causal_classifier": "Classify causal vs predictive research questions",
            "model_selector": "Select appropriate statistical/ML models",
            "evaluation_designer": "Design rigorous evaluation frameworks",
            "dowhy_planner": "Plan causal inference using DoWhy",
            "baseline_generator": "Generate appropriate baseline models",
            "benchmark_finder": "Find state-of-the-art benchmarks",
            "simulation_designer": "Design numerical simulations"
        }
        
        self.logger = logging.getLogger(f"Methodologist.{agent_id}")
    
    async def receive_task(self, task: Task) -> None:
        """Receive and process a task"""
        self.active_tasks[task.task_id] = task
        self.logger.info(f"Methodologist received task: {task.goal}")
        
        # Execute the ReAct loop
        result = await self.react_loop(task)
        await self.return_result(task.task_id, result)
    
    async def report_status(self, task_id: str, status: TaskStatus) -> None:
        """Report task status to MCP"""
        if self.mcp_hub:
            pass  # Update MCP hub
        self.logger.info(f"Task {task_id} status: {status.value}")
    
    async def return_result(self, task_id: str, result: TaskResult) -> None:
        """Return completed result to MCP"""
        if self.mcp_hub:
            await self.mcp_hub.publish_result(result)
        self.logger.info(f"Methodologist returned result for task {task_id}")
    
    async def request_tool_use(self, tool_request: ToolRequest) -> Dict[str, Any]:
        """Request tool use through MCP"""
        if self.mcp_hub:
            return await self.mcp_hub.request_tool_use(tool_request)
        return await self.simulate_tool_use(tool_request)
    
    async def think(self, observation: str, context: Dict[str, Any]) -> str:
        """Methodologist-specific reasoning"""
        hypothesis = context.get("hypothesis", {})
        data_sources = context.get("data_sources", {})
        
        thinking_prompt = f"""
        As a Methodologist agent, I need to design rigorous experimental methodology.
        
        Current situation: {observation}
        Hypothesis: {hypothesis}
        Available data: {data_sources}
        
        My methodological analysis:
        1. How should I formalize this hypothesis?  
        2. Is this a causal or predictive question?
        3. What models are appropriate?
        4. How should I evaluate the approach?
        5. What baselines and benchmarks are needed?
        
        Current thinking:
        """
        
        if self.llm_client:
            response = await self._query_llm(thinking_prompt)
            return response
        
        # Mock thinking for testing
        return f"""
        Analyzing hypothesis: {hypothesis}
        
        Key considerations:
        1. This appears to be a causal research question
        2. Need to formalize variables and relationships
        3. Require appropriate causal inference methods
        4. Must design rigorous evaluation framework
        
        Next action: Classify question type and formalize hypothesis.
        """
    
    async def act(self, thought: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Methodologist-specific actions"""
        
        if "formalize" in thought.lower() or "ontology" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "obi_ontology",
                "parameters": {
                    "hypothesis": context.get("hypothesis", {}),
                    "variables": context.get("hypothesis", {}).get("variables", {})
                },
                "reasoning": "Formalizing hypothesis using OBI ontology"
            }
        elif "causal" in thought.lower() or "predictive" in thought.lower() or "classify" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "causal_classifier",
                "parameters": {
                    "research_question": context.get("hypothesis", {}).get("primary_hypothesis", ""),
                    "context": context
                },
                "reasoning": "Classifying research question as causal vs predictive"
            }
        elif "model" in thought.lower() and "select" in thought.lower():
            return {
                "action_type": "use_tool", 
                "tool_name": "model_selector",
                "parameters": {
                    "task_type": context.get("task_type", ""),
                    "variables": context.get("hypothesis", {}).get("variables", {}),
                    "data_characteristics": context.get("data_sources", {})
                },
                "reasoning": "Selecting appropriate statistical/ML models"
            }
        elif "evaluation" in thought.lower() or "framework" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "evaluation_designer",
                "parameters": {
                    "task_type": context.get("task_type", ""),
                    "proposed_model": context.get("proposed_model", {}),
                    "hypothesis": context.get("hypothesis", {})
                },
                "reasoning": "Designing comprehensive evaluation framework"
            }
        elif "baseline" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "baseline_generator", 
                "parameters": {
                    "task_type": context.get("task_type", ""),
                    "variables": context.get("hypothesis", {}).get("variables", {})
                },
                "reasoning": "Generating appropriate baseline models"
            }
        elif "benchmark" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "benchmark_finder",
                "parameters": {
                    "research_domain": context.get("research_domain", ""),
                    "task_type": context.get("task_type", "")
                },
                "reasoning": "Finding state-of-the-art benchmark methods"
            }
        else:
            return {
                "action_type": "complete",
                "reasoning": "Methodology design complete"
            }
    
    async def simulate_tool_use(self, tool_request: ToolRequest) -> Dict[str, Any]:
        """Simulate tool usage for testing"""
        tool_name = tool_request.tool_name
        parameters = tool_request.parameters
        
        if tool_name == "obi_ontology":
            # Simulate OBI ontology formalization
            hypothesis = parameters.get("hypothesis", {})
            variables = parameters.get("variables", {})
            
            return {
                "success": True,
                "result": {
                    "formalized_hypothesis": {
                        "study_design": "observational_study",
                        "intervention": variables.get("independent", ""),
                        "outcome": variables.get("dependent", ""),
                        "moderator": variables.get("moderator", ""),
                        "confounders": variables.get("controls", []),
                        "population": "target_population",
                        "temporal_scope": "cross_sectional"
                    },
                    "obi_mapping": {
                        "study_design": "OBI_0000066",
                        "intervention": "OBI_0000115", 
                        "outcome_measure": "OBI_0000070",
                        "study_population": "OBI_0000181"
                    },
                    "machine_readable_format": "JSON-LD with OBI URIs"
                }
            }
        
        elif tool_name == "causal_classifier":
            # Simulate causal vs predictive classification
            research_question = parameters.get("research_question", "")
            
            # Simple heuristics for classification
            is_causal = any(word in research_question.lower() for word in 
                          ["cause", "effect", "impact", "influence", "affect", "causal"])
            
            return {
                "success": True,
                "result": {
                    "classification": "causal_inference" if is_causal else "prediction",
                    "confidence": 0.87 if is_causal else 0.82,
                    "reasoning": [
                        "Research question contains causal language" if is_causal else "Focus appears to be on prediction",
                        "Variables suggest causal relationship" if is_causal else "Variables suggest predictive modeling",
                        "Goal is inference, not prediction" if is_causal else "Goal is prediction accuracy"
                    ],
                    "recommended_approach": "causal_inference_methods" if is_causal else "machine_learning_methods",
                    "key_considerations": [
                        "Need to address confounding" if is_causal else "Focus on predictive accuracy",
                        "Causal identification required" if is_causal else "Feature engineering important",
                        "Effect size estimation" if is_causal else "Model generalization"
                    ]
                }
            }
        
        elif tool_name == "model_selector":
            # Simulate model selection
            task_type = parameters.get("task_type", "")
            variables = parameters.get("variables", {})
            
            if task_type == "causal_inference":
                models = {
                    "proposed_model": {
                        "name": "Statsmodels OLS with Interaction Term",
                        "formula": f"{variables.get('dependent', 'Y')} ~ {variables.get('independent', 'X')} + {variables.get('moderator', 'Z')} + {variables.get('independent', 'X')}*{variables.get('moderator', 'Z')} + age + gender",
                        "library": "statsmodels",
                        "method": "ordinary_least_squares"
                    },
                    "alternative_models": [
                        {
                            "name": "DoWhy Causal Inference",
                            "method": "backdoor_adjustment",
                            "library": "dowhy"
                        },
                        {
                            "name": "Propensity Score Matching",
                            "method": "matching",
                            "library": "causalml"
                        }
                    ]
                }
            else:
                models = {
                    "proposed_model": {
                        "name": "XGBoost Classifier",
                        "library": "xgboost",
                        "method": "gradient_boosting"
                    },
                    "alternative_models": [
                        {
                            "name": "Random Forest",
                            "library": "scikit-learn",
                            "method": "ensemble"
                        },
                        {
                            "name": "Logistic Regression",
                            "library": "scikit-learn", 
                            "method": "linear_model"
                        }
                    ]
                }
            
            return {
                "success": True,
                "result": models
            }
        
        elif tool_name == "evaluation_designer":
            # Simulate evaluation framework design
            task_type = parameters.get("task_type", "")
            
            if task_type == "causal_inference":
                framework = {
                    "evaluation_strategy": [
                        {
                            "type": "real_world_data",
                            "metrics": ["p_value", "effect_size_CI", "R_squared"],
                            "validation": "DoWhy refutation checks (placebo_treatment, random_confounder)"
                        },
                        {
                            "type": "numerical_simulation", 
                            "description": "Simulate data with known interaction effect (beta=0.25)",
                            "conditions": ["N=1000_low_noise", "N=1000_high_noise", "N=5000_high_noise"]
                        }
                    ],
                    "robustness_checks": [
                        "Add random confounder",
                        "Replace treatment with placebo",
                        "Subset refutation",
                        "Bootstrap validation"
                    ],
                    "comparison_metrics": "Bias and variance of estimated interaction effect vs ground-truth"
                }
            else:
                framework = {
                    "evaluation_strategy": [
                        {
                            "type": "cross_validation",
                            "method": "k_fold",
                            "k": 5,
                            "metrics": ["accuracy", "f1_score", "auc_roc"]
                        },
                        {
                            "type": "holdout_validation",
                            "train_size": 0.7,
                            "validation_size": 0.15,
                            "test_size": 0.15
                        }
                    ],
                    "performance_metrics": ["precision", "recall", "f1_score", "accuracy"],
                    "comparison_metrics": "Predictive accuracy vs baseline and benchmark"
                }
            
            return {
                "success": True,
                "result": framework
            }
        
        elif tool_name == "baseline_generator":
            # Simulate baseline model generation
            task_type = parameters.get("task_type", "")
            variables = parameters.get("variables", {})
            
            if task_type == "causal_inference":
                baseline = {
                    "name": "Simple OLS Regression",
                    "formula": f"{variables.get('dependent', 'Y')} ~ {variables.get('independent', 'X')}",
                    "description": "Basic linear model without controls or interactions",
                    "purpose": "Establish minimum performance threshold for causal estimation"
                }
            else:
                baseline = {
                    "name": "Predict the Mean",
                    "method": "always_predict_majority_class",
                    "description": "Naive baseline that always predicts the most common class",
                    "purpose": "Establish minimum accuracy threshold"
                }
            
            return {
                "success": True,
                "result": {
                    "baseline_model": baseline,
                    "expected_performance": "Lower bound for model comparison"
                }
            }
        
        elif tool_name == "benchmark_finder":
            # Simulate benchmark finding
            research_domain = parameters.get("research_domain", "")
            task_type = parameters.get("task_type", "")
            
            return {
                "success": True,
                "result": {
                    "benchmark_method": {
                        "name": f"State-of-the-art method for {research_domain}",
                        "paper": "Benchmark Paper et al. (2023)",
                        "method": "Advanced statistical/ML approach",
                        "performance": "Reported benchmark performance metrics",
                        "implementation": "Available in standard libraries"
                    },
                    "comparison_approach": "Compare proposed method against this benchmark",
                    "evaluation_criteria": "Statistical significance and effect size (causal) or predictive accuracy (ML)"
                }
            }
        
        elif tool_name == "dowhy_planner":
            # Simulate DoWhy causal inference planning
            return {
                "success": True,
                "result": {
                    "causal_workflow": {
                        "step_1_model": "Construct causal graph with confounders",
                        "step_2_identify": "Use backdoor criterion for identification", 
                        "step_3_estimate": "Linear regression with confounders",
                        "step_4_refute": "Placebo treatment and random confounder tests"
                    },
                    "causal_assumptions": [
                        "No unobserved confounders",
                        "Correct causal graph structure",
                        "Linear relationships"
                    ],
                    "identification_strategy": "Backdoor adjustment with controls"
                }
            }
        
        elif tool_name == "simulation_designer":
            # Simulate numerical simulation design
            return {
                "success": True,
                "result": {
                    "simulation_plan": {
                        "data_generation": "Generate synthetic data with known ground truth",
                        "sample_sizes": [1000, 5000, 10000],
                        "noise_levels": ["low", "medium", "high"],
                        "effect_sizes": [0.1, 0.25, 0.5],
                        "confounding_levels": ["none", "weak", "strong"]
                    },
                    "evaluation_metrics": "Bias, variance, and MSE of effect estimates",
                    "ground_truth": "Known causal effect to compare against"
                }
            }
        
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    async def design_comprehensive_methodology(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Design comprehensive methodology from scratch"""
        
        # Step 1: Formalize hypothesis
        formalization = await self.formalize_hypothesis(context.get("hypothesis", {}))
        
        # Step 2: Classify task type
        classification = await self.classify_research_task(context.get("hypothesis", {}))
        
        # Step 3: Select models
        model_selection = await self.select_models(classification, context)
        
        # Step 4: Design evaluation
        evaluation_plan = await self.design_evaluation_framework(classification, model_selection)
        
        # Step 5: Generate comparison framework
        comparison_plan = await self.design_comparison_framework(classification, context)
        
        return {
            "formalized_hypothesis": formalization,
            "task_classification": classification,
            "model_selection": model_selection,
            "evaluation_plan": evaluation_plan,
            "comparison_framework": comparison_plan,
            "methodology_confidence": 0.89
        }
    
    async def formalize_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Formalize hypothesis using OBI ontology"""
        tool_request = ToolRequest(
            task_id="formalize",
            originator=self.agent_id,
            tool_name="obi_ontology",
            parameters={"hypothesis": hypothesis}
        )
        result = await self.simulate_tool_use(tool_request)
        return result.get("result", {})
    
    async def classify_research_task(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Classify as causal vs predictive task"""
        tool_request = ToolRequest(
            task_id="classify",
            originator=self.agent_id,
            tool_name="causal_classifier",
            parameters={"research_question": hypothesis.get("primary_hypothesis", "")}
        )
        result = await self.simulate_tool_use(tool_request)
        return result.get("result", {})
    
    async def select_models(self, classification: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate models based on classification"""
        tool_request = ToolRequest(
            task_id="select_models",
            originator=self.agent_id,
            tool_name="model_selector",
            parameters={
                "task_type": classification.get("classification", ""),
                "variables": context.get("hypothesis", {}).get("variables", {}),
                "data_characteristics": context.get("data_sources", {})
            }
        )
        result = await self.simulate_tool_use(tool_request)
        return result.get("result", {})
    
    async def design_evaluation_framework(self, classification: Dict[str, Any], model_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Design comprehensive evaluation framework"""
        tool_request = ToolRequest(
            task_id="design_eval",
            originator=self.agent_id,
            tool_name="evaluation_designer",
            parameters={
                "task_type": classification.get("classification", ""),
                "proposed_model": model_selection.get("proposed_model", {})
            }
        )
        result = await self.simulate_tool_use(tool_request)
        return result.get("result", {})
    
    async def design_comparison_framework(self, classification: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Design baseline and benchmark comparison framework"""
        
        # Get baseline
        baseline_request = ToolRequest(
            task_id="baseline",
            originator=self.agent_id,
            tool_name="baseline_generator",
            parameters={
                "task_type": classification.get("classification", ""),
                "variables": context.get("hypothesis", {}).get("variables", {})
            }
        )
        baseline_result = await self.simulate_tool_use(baseline_request)
        
        # Get benchmark
        benchmark_request = ToolRequest(
            task_id="benchmark",
            originator=self.agent_id,
            tool_name="benchmark_finder",
            parameters={
                "research_domain": context.get("research_domain", ""),
                "task_type": classification.get("classification", "")
            }
        )
        benchmark_result = await self.simulate_tool_use(benchmark_request)
        
        return {
            "baseline": baseline_result.get("result", {}),
            "benchmark": benchmark_result.get("result", {}),
            "comparison_strategy": "Three-way comparison: Proposed vs Baseline vs Benchmark"
        }


__all__ = ["MethodologistAgent"]