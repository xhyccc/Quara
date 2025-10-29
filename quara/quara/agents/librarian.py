"""
Librarian Agent - Data Collection and Domain Knowledge Curation
"""

import asyncio
from typing import Dict, Any, List, Optional
import logging

from ..core.base import (
    BaseReActAgent, AgentRole, Task, TaskResult, TaskStatus, ToolRequest
)


class LibrarianAgent(BaseReActAgent):
    """
    The Librarian Agent handles data and knowledge curation.
    
    Core functions:
    - Domain knowledge injection via web search and APIs
    - Academic literature retrieval (PubMed, ArXiv, Scopus)
    - Quantitative data retrieval (Kaggle, UCI, World Bank)
    - SoAy methodology for complex API sequences
    """
    
    def __init__(self, agent_id: str = "librarian", llm_client=None, mcp_hub=None):
        super().__init__(agent_id, AgentRole.LIBRARIAN, llm_client)
        
        self.mcp_hub = mcp_hub
        self.retrieved_data = {}
        self.api_cache = {}
        
        # Specialized tools for Librarian
        self.tools = {
            "web_search": "General web search using DuckDuckGo",
            "pubmed_api": "Search PubMed for biomedical literature",
            "arxiv_api": "Search ArXiv for academic preprints",
            "scopus_api": "Search Scopus database",
            "kaggle_api": "Search and retrieve Kaggle datasets",
            "uci_api": "Search UCI Machine Learning Repository",
            "worldbank_api": "World Bank data API",
            "openapi_generic": "Generic OpenAPI interface",
            "soay_solver": "Solution-based API query sequences"
        }
        
        self.logger = logging.getLogger(f"Librarian.{agent_id}")
    
    async def receive_task(self, task: Task) -> None:
        """Receive and process a task"""
        self.active_tasks[task.task_id] = task
        self.logger.info(f"Librarian received task: {task.goal}")
        
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
        self.logger.info(f"Librarian returned result for task {task_id}")
    
    async def request_tool_use(self, tool_request: ToolRequest) -> Dict[str, Any]:
        """Request tool use through MCP"""
        if self.mcp_hub:
            return await self.mcp_hub.request_tool_use(tool_request)
        return await self.simulate_tool_use(tool_request)
    
    async def think(self, observation: str, context: Dict[str, Any]) -> str:
        """Librarian-specific reasoning"""
        hypothesis = context.get("hypothesis", {})
        search_type = context.get("type", "comprehensive")
        
        thinking_prompt = f"""
        As a Librarian agent, I need to gather comprehensive data and knowledge.
        
        Current situation: {observation}
        Hypothesis: {hypothesis}
        Search type: {search_type}
        
        My retrieval strategy:
        1. What type of information do I need?
        2. Which APIs/sources are most relevant?
        3. What search terms should I use?
        4. How can I ensure comprehensive coverage?
        5. What data formats are needed?
        
        Current thinking:
        """
        
        if self.llm_client:
            response = await self._query_llm(thinking_prompt)
            return response
        
        # Mock thinking for testing
        return f"""
        I need to gather information for: {hypothesis}
        
        My approach:
        1. Start with academic literature search
        2. Find relevant datasets
        3. Gather domain knowledge
        4. Provide structured results
        
        Next action: Begin literature search.
        """
    
    async def act(self, thought: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Librarian-specific actions"""
        
        if "literature search" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "pubmed_api",
                "parameters": {
                    "query": self.extract_search_terms(context),
                    "max_results": 100
                },
                "reasoning": "Starting with biomedical literature search"
            }
        elif "dataset" in thought.lower() or "data" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "kaggle_api", 
                "parameters": {
                    "query": self.extract_search_terms(context),
                    "dataset_type": "csv"
                },
                "reasoning": "Searching for relevant datasets"
            }
        elif "domain knowledge" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "web_search",
                "parameters": {
                    "query": self.extract_search_terms(context),
                    "num_results": 20
                },
                "reasoning": "Gathering domain-specific knowledge"
            }
        elif "comprehensive" in thought.lower():
            return {
                "action_type": "use_tool",
                "tool_name": "soay_solver",
                "parameters": {
                    "query_type": "multi_source_search",
                    "topic": context.get("hypothesis", {})
                },
                "reasoning": "Using SoAy for comprehensive multi-source search"
            }
        else:
            return {
                "action_type": "complete",
                "reasoning": "Data collection complete"
            }
    
    def extract_search_terms(self, context: Dict[str, Any]) -> str:
        """Extract appropriate search terms from context"""
        hypothesis = context.get("hypothesis", {})
        
        if isinstance(hypothesis, dict):
            terms = []
            if "variables" in hypothesis:
                vars_dict = hypothesis["variables"]
                terms.extend([
                    vars_dict.get("independent", ""),
                    vars_dict.get("dependent", ""), 
                    vars_dict.get("moderator", "")
                ])
            
            if "primary_hypothesis" in hypothesis:
                # Extract key terms from hypothesis text
                terms.append(hypothesis["primary_hypothesis"])
            
            return " ".join(filter(None, terms))
        
        return str(hypothesis)
    
    async def simulate_tool_use(self, tool_request: ToolRequest) -> Dict[str, Any]:
        """Simulate tool usage for testing"""
        tool_name = tool_request.tool_name
        parameters = tool_request.parameters
        
        if tool_name == "pubmed_api":
            # Simulate PubMed search
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 100)
            
            mock_papers = [
                {
                    "pmid": "12345678",
                    "title": f"Clinical study of {query} effects",
                    "authors": ["Smith J", "Jones A"],
                    "abstract": f"This randomized controlled trial examined {query}...",
                    "journal": "Journal of Medical Research",
                    "year": 2023,
                    "doi": "10.1000/example",
                    "mesh_terms": ["Term1", "Term2", "Term3"]
                },
                {
                    "pmid": "87654321", 
                    "title": f"Meta-analysis of {query} interventions",
                    "authors": ["Brown K", "Wilson M"],
                    "abstract": f"Systematic review and meta-analysis of {query}...",
                    "journal": "Systematic Reviews",
                    "year": 2022,
                    "doi": "10.1000/example2",
                    "mesh_terms": ["Term2", "Term4", "Term5"]
                }
            ]
            
            return {
                "success": True,
                "result": {
                    "papers": mock_papers,
                    "total_found": len(mock_papers),
                    "search_query": query,
                    "source": "PubMed"
                }
            }
        
        elif tool_name == "kaggle_api":
            # Simulate Kaggle dataset search
            query = parameters.get("query", "")
            
            mock_datasets = [
                {
                    "id": "dataset_123",
                    "title": f"{query} Dataset",
                    "description": f"Comprehensive dataset for {query} research",
                    "size": "50MB",
                    "format": "CSV",
                    "rows": 10000,
                    "columns": 25,
                    "url": "https://kaggle.com/datasets/dataset_123",
                    "license": "CC BY-SA 4.0",
                    "last_updated": "2023-10-01"
                },
                {
                    "id": "dataset_456",
                    "title": f"Longitudinal {query} Study",
                    "description": f"10-year longitudinal data on {query}",
                    "size": "120MB", 
                    "format": "CSV",
                    "rows": 25000,
                    "columns": 40,
                    "url": "https://kaggle.com/datasets/dataset_456",
                    "license": "Open Database",
                    "last_updated": "2023-09-15"
                }
            ]
            
            return {
                "success": True,
                "result": {
                    "datasets": mock_datasets,
                    "total_found": len(mock_datasets),
                    "search_query": query,
                    "source": "Kaggle"
                }
            }
        
        elif tool_name == "web_search":
            # Simulate web search
            query = parameters.get("query", "")
            num_results = parameters.get("num_results", 10)
            
            mock_results = [
                {
                    "title": f"Expert Guide to {query}",
                    "url": f"https://example.com/{query.replace(' ', '-')}",
                    "snippet": f"Comprehensive overview of {query} including key concepts...",
                    "source": "Academic Institution"
                },
                {
                    "title": f"{query} Research Methods",
                    "url": f"https://research.org/{query}",
                    "snippet": f"Methodological approaches for studying {query}...",
                    "source": "Research Organization"
                },
                {
                    "title": f"Statistical Analysis of {query}",
                    "url": f"https://stats.edu/analysis/{query}",
                    "snippet": f"Statistical methods and considerations for {query} analysis...",
                    "source": "Statistical Department"
                }
            ]
            
            return {
                "success": True,
                "result": {
                    "results": mock_results[:num_results],
                    "total_found": len(mock_results),
                    "search_query": query,
                    "source": "Web Search"
                }
            }
        
        elif tool_name == "soay_solver":
            # Simulate SoAy comprehensive search
            query_type = parameters.get("query_type", "")
            topic = parameters.get("topic", {})
            
            return {
                "success": True,
                "result": {
                    "search_strategy": "Multi-source comprehensive search",
                    "sources_queried": [
                        "PubMed (biomedical literature)",
                        "ArXiv (preprints)",
                        "Kaggle (datasets)",
                        "UCI Repository (ML datasets)",
                        "Web search (domain knowledge)"
                    ],
                    "total_papers": 150,
                    "total_datasets": 12,
                    "domain_knowledge_items": 45,
                    "cross_references": 23,
                    "confidence_score": 0.92,
                    "completeness_score": 0.87
                }
            }
        
        elif tool_name == "arxiv_api":
            # Simulate ArXiv search
            query = parameters.get("query", "")
            
            mock_preprints = [
                {
                    "id": "2310.12345",
                    "title": f"Novel Approaches to {query}",
                    "authors": ["Author X", "Author Y"],
                    "abstract": f"We present new methods for analyzing {query}...",
                    "categories": ["stat.ML", "cs.LG"],
                    "submitted": "2023-10-15",
                    "updated": "2023-10-20"
                }
            ]
            
            return {
                "success": True,
                "result": {
                    "preprints": mock_preprints,
                    "total_found": len(mock_preprints),
                    "search_query": query,
                    "source": "ArXiv"
                }
            }
        
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    async def comprehensive_search(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive multi-source search"""
        
        search_terms = self.extract_search_terms({"hypothesis": topic})
        
        # Search academic literature
        literature_results = []
        for api in ["pubmed_api", "arxiv_api"]:
            tool_request = ToolRequest(
                task_id="search",
                originator=self.agent_id,
                tool_name=api,
                parameters={"query": search_terms}
            )
            result = await self.simulate_tool_use(tool_request)
            if result.get("success"):
                literature_results.append(result["result"])
        
        # Search datasets
        dataset_results = []
        for api in ["kaggle_api", "uci_api"]:
            tool_request = ToolRequest(
                task_id="search",
                originator=self.agent_id,
                tool_name=api,
                parameters={"query": search_terms}
            )
            result = await self.simulate_tool_use(tool_request)
            if result.get("success"):
                dataset_results.append(result["result"])
        
        # Web search for domain knowledge
        web_request = ToolRequest(
            task_id="search",
            originator=self.agent_id,
            tool_name="web_search",
            parameters={"query": search_terms, "num_results": 20}
        )
        web_result = await self.simulate_tool_use(web_request)
        
        return {
            "literature": literature_results,
            "datasets": dataset_results,
            "domain_knowledge": web_result.get("result", {}),
            "search_terms": search_terms,
            "total_sources": len(literature_results) + len(dataset_results) + 1
        }
    
    async def apply_soay_methodology(self, complex_query: str) -> Dict[str, Any]:
        """Apply SoAy (Solution-based) methodology for complex API sequences"""
        
        # SoAy breaks complex queries into executable API sequences
        query_analysis = await self.analyze_query_complexity(complex_query)
        
        if query_analysis["complexity"] == "high":
            # Decompose into sub-queries
            sub_queries = await self.decompose_query(complex_query)
            
            results = []
            for sub_query in sub_queries:
                sub_result = await self.execute_api_sequence(sub_query)
                results.append(sub_result)
            
            # Combine results
            combined_result = await self.combine_sub_results(results)
            return combined_result
        else:
            # Simple query, direct execution
            return await self.execute_simple_query(complex_query)
    
    async def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity to determine SoAy approach"""
        # Mock analysis
        return {
            "complexity": "high" if len(query.split()) > 10 else "low",
            "api_count": 3,
            "sequence_length": 5
        }
    
    async def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        # Mock decomposition
        return [
            f"Sub-query 1 from: {query}",
            f"Sub-query 2 from: {query}",
            f"Sub-query 3 from: {query}"
        ]
    
    async def execute_api_sequence(self, sub_query: str) -> Dict[str, Any]:
        """Execute API sequence for sub-query"""
        # Mock execution
        return {
            "sub_query": sub_query,
            "results": f"Results for {sub_query}",
            "api_calls": 3
        }
    
    async def combine_sub_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from sub-queries"""
        return {
            "combined_results": results,
            "total_sub_queries": len(results),
            "synthesis": "Combined comprehensive results"
        }
    
    async def execute_simple_query(self, query: str) -> Dict[str, Any]:
        """Execute simple query directly"""
        return {
            "query": query,
            "approach": "direct",
            "result": f"Direct result for {query}"
        }


__all__ = ["LibrarianAgent"]