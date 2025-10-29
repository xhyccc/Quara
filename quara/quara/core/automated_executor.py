"""
Automated Research Executor - Full workflow automation
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

from ..utils import create_llm_client, ModelPurpose
from ..tools import (
    DataDownloader, DataAnalyzer, CodeExecutor, VisualizationGenerator,
    WebSearchTool, WebContentFetcher, FileSaver, FileSearcher
)


class AutomatedResearchExecutor:
    """Fully automated research execution with data download, code execution, and visualization"""
    
    def __init__(self, project_root: str = "./research_projects"):
        self.project_root = Path(project_root)
        self.project_root.mkdir(parents=True, exist_ok=True)
        
        self.llm_client = create_llm_client()
        self.data_downloader = DataDownloader()
        self.data_analyzer = DataAnalyzer()
        self.code_executor = CodeExecutor()
        self.viz_generator = VisualizationGenerator()
        self.web_search = WebSearchTool()
        self.web_fetcher = WebContentFetcher()
        self.file_saver = FileSaver()
        self.file_searcher = FileSearcher()
        
        self.logger = logging.getLogger(__name__)
        self.debug_mode = self.logger.isEnabledFor(logging.DEBUG)
    
    async def execute_research(self, research_question: str) -> Dict[str, Any]:
        """
        Execute full automated research workflow
        
        Args:
            research_question: The research question to investigate
            
        Returns:
            Dictionary with all research outputs
        """
        # Create project directory
        project_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_dir = self.project_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting automated research: {research_question}")
        self.logger.info(f"Project directory: {project_dir}")
        
        if self.debug_mode:
            self.logger.debug(f"DEBUG MODE ENABLED")
            self.logger.debug(f"Project ID: {project_id}")
            self.logger.debug(f"Full project path: {project_dir.absolute()}")
        
        artifacts = {
            "project_id": project_id,
            "question": research_question,
            "directory": str(project_dir),
            "started_at": datetime.now().isoformat(),
            "stages": {}
        }
        
        try:
            # Stage 1: Research Planning
            self.logger.info("Stage 1: Research Planning")
            if self.debug_mode:
                self.logger.debug(f"Planning research for question: {research_question}")
            plan = await self._plan_research(research_question)
            artifacts["stages"]["planning"] = plan
            
            # Save grounding context to markdown file
            if "grounding_context" in plan:
                grounding_file = project_dir / "grounding_context.md"
                with open(grounding_file, 'w', encoding='utf-8') as f:
                    f.write(plan["grounding_context"])
                self.logger.info(f"📄 Saved grounding context to: grounding_context.md")
                if self.debug_mode:
                    self.logger.debug(f"Grounding context file: {grounding_file}")
            
            if self.debug_mode:
                self.logger.debug(f"Plan generated: {json.dumps(plan, indent=2, default=str)[:500]}...")
            
            # Stage 2: Data Collection
            self.logger.info("Stage 2: Data Collection")
            if self.debug_mode:
                self.logger.debug(f"Starting data collection with plan: {plan.get('data_sources', 'N/A')}")
            data_results = await self._collect_data(plan, project_dir)
            artifacts["stages"]["data_collection"] = data_results
            if self.debug_mode:
                self.logger.debug(f"Data collected: {len(data_results.get('datasets', {}))} datasets")
                for ticker, info in data_results.get('datasets', {}).items():
                    self.logger.debug(f"  - {ticker}: {info.get('rows')} rows, {info.get('date_range')}")
            
            # Stage 3: Code Generation & Execution
            self.logger.info("Stage 3: Analysis Code Generation & Execution")
            if self.debug_mode:
                self.logger.debug(f"Generating analysis code for {len(data_results.get('datasets', {}))} datasets")
            
            # Include grounding context in analysis
            grounding_context = plan.get("grounding_context", "")
            analysis_results = await self._generate_and_execute_analysis(
                research_question, data_results, project_dir, grounding_context
            )
            artifacts["stages"]["analysis"] = analysis_results
            if self.debug_mode:
                self.logger.debug(f"Analysis execution success: {analysis_results['execution']['success']}")
                self.logger.debug(f"Analysis stdout: {analysis_results['execution']['stdout'][:300]}...")
            
            # Stage 4: Basic Visualization
            self.logger.info("Stage 4: Basic Visualization Generation")
            if self.debug_mode:
                self.logger.debug("Starting basic visualization generation with LLM")
            
            # Include grounding context in visualizations
            grounding_context = plan.get("grounding_context", "")
            viz_results = await self._generate_visualizations(
                analysis_results, project_dir, grounding_context
            )
            artifacts["stages"]["visualizations"] = viz_results
            if self.debug_mode:
                self.logger.debug(f"Basic viz execution success: {viz_results['execution']['success']}")
                self.logger.debug(f"Basic viz plots generated: {len(viz_results.get('plots', []))}")
                self.logger.debug(f"Basic viz stdout: {viz_results['execution']['output']}")
            
            # Stage 5: Report Writing
            self.logger.info("Stage 5: Research Report Generation")
            if self.debug_mode:
                self.logger.debug("Generating research report with LLM and grounding context")
            
            # Include grounding context in report
            grounding_context = plan.get("grounding_context", "")
            report = await self._generate_report(
                research_question, artifacts, project_dir, grounding_context
            )
            artifacts["stages"]["report"] = report
            if self.debug_mode:
                self.logger.debug(f"Report file: {report['file']}")
                self.logger.debug(f"Report preview: {report['preview'][:200]}...")
            
            # Stage 6: Comprehensive Section-Specific Visualizations
            self.logger.info("Stage 6: Generating Section-Specific Visualizations")
            if self.debug_mode:
                self.logger.debug("Starting section-specific visualization generation with LLM and grounding context")
            
            # Include grounding context in section visualizations
            grounding_context = plan.get("grounding_context", "")
            enhanced_viz_results = await self._generate_section_visualizations(
                research_question, report, data_results, project_dir, grounding_context
            )
            artifacts["stages"]["section_visualizations"] = enhanced_viz_results
            if self.debug_mode:
                self.logger.debug(f"Section viz execution success: {enhanced_viz_results['execution']['success']}")
                self.logger.debug(f"Section viz plots generated: {len(enhanced_viz_results.get('plots', []))}")
                self.logger.debug(f"Section viz stdout: {enhanced_viz_results['execution']['output']}")
                if enhanced_viz_results['execution'].get('error'):
                    self.logger.debug(f"Section viz stderr: {enhanced_viz_results['execution']['error']}")
            
            # Save metadata
            artifacts["completed_at"] = datetime.now().isoformat()
            artifacts["success"] = True
            
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(artifacts, f, indent=2, default=str)
            
            self.logger.info(f"Research completed successfully: {project_id}")
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            artifacts["error"] = str(e)
            artifacts["success"] = False
            return artifacts
    
    async def _plan_research(self, question: str) -> Dict[str, Any]:
        """Generate research plan with data requirements and web search grounding"""
        
        # Stage 1: Comprehensive web search with multiple reformulated queries
        self.logger.info("🔍 Performing comprehensive web search for research grounding...")
        if self.debug_mode:
            self.logger.debug(f"Original research question: {question}")
        
        # Generate multiple search queries using LLM
        now = datetime.now()
        current_datetime = now.strftime('%Y-%m-%d %H:%M:%S')
        weekday = now.strftime('%A')
        month = now.strftime('%B')
        quarter = (now.month - 1) // 3 + 1
        
        temporal_context = f"""Current Date and Time: {current_datetime}
Weekday: {weekday}
Date: {now.day}
Month: {month}
Quarter: Q{quarter}
Year: {now.year}"""
        
        query_generation_prompt = f"""{temporal_context}

Given this research question, generate 5 different search queries that would help gather comprehensive background information.

Research Question: {question}

Generate queries that cover:
1. The main research question (direct)
2. Recent developments and news (use current year {now.year}, Q{quarter})
3. Most recent developments and news (use today {now.day} {month} {now.year})
4. Real-time information and trends (use current time {now})
5. Academic research and methodologies
6. Data sources and datasets
7. Related factors and context

Return as JSON array: {{"queries": ["query1", "query2", ...]}}"""

        if self.debug_mode:
            self.logger.debug("Generating search queries with LLM...")
            self.logger.debug(f"Temporal context:\n{temporal_context}")
        
        queries_response = await self.llm_client.generate(
            query_generation_prompt,
            temperature=0.5,
            purpose=ModelPurpose.REASONING
        )
        
        # Extract queries from response
        import re
        search_queries = []
        try:
            json_match = re.search(r'\{.*\}', queries_response, re.DOTALL)
            if json_match:
                queries_data = json.loads(json_match.group())
                search_queries = queries_data.get('queries', [])
                if self.debug_mode:
                    self.logger.debug(f"Generated {len(search_queries)} search queries: {search_queries}")
        except:
            # Fallback to default queries
            current_year = datetime.now().year
            now = datetime.now()
            current_datetime = now.strftime('%Y-%m-%d %H:%M:%S')
            weekday = now.strftime('%A')
            month = now.strftime('%B')
            quarter = (now.month - 1) // 3 + 1
            search_queries = [
                question,
                f"{question} recent research {current_year}",
                f"{question} recent research {current_year} Q{quarter}",
                f"{question} recent research {month}/{current_year}",
                f"{question} data analysis methods",
                f"{question} data analysis methods {current_year}",
                f"{question} data analysis methods {current_year} Q{quarter}",
                f"{question} data analysis methods {month}/{current_year}",
                f"{question} academic studies",
                f"{question} academic studies {current_year}",
                f"{question} academic studies {current_year} Q{quarter}",
                f"{question} academic studies {month}/{current_year}",
                f"{question} trends and insights {current_year}",
                f"{question} trends and insights {current_year} Q{quarter}",
                f"{question} trends and insights {month}/{current_year}",
                f"{question} most recent news and trending {current_datetime}"
            ]
            if self.debug_mode:
                self.logger.debug("Using fallback search queries")
        
        # Perform web searches for each query
        all_search_results = []
        search_summaries = []
        
        for i, query in enumerate(search_queries[:5], 1):  # Limit to 5 searches
            self.logger.info(f"🔎 Search {i}/5: {query[:60]}...")
            if self.debug_mode:
                self.logger.debug(f"Executing web search: {query}")
            
            result = await self.web_search.search(query, max_results=8)
            if result["success"]:
                results = result["results"]
                all_search_results.extend(results)
                
                # Create summary for this query
                query_summary = f"\n### Query {i}: {query}\n"
                query_summary += f"Found {len(results)} results:\n\n"
                for j, r in enumerate(results[:5], 1):
                    query_summary += f"{j}. **{r['title']}**\n"
                    query_summary += f"   {r['snippet'][:150]}...\n"
                    query_summary += f"   Source: {r['url']}\n\n"
                
                search_summaries.append(query_summary)
                
                if self.debug_mode:
                    self.logger.debug(f"Query '{query}' returned {len(results)} results")
            else:
                if self.debug_mode:
                    self.logger.debug(f"Search failed for query: {query}")
        
        # Create comprehensive grounding context markdown
        grounding_context = f"""# Research Grounding Context

## Research Question
{question}

## Search Summary
Generated {len(search_queries)} search queries and found {len(all_search_results)} total results.

## Detailed Search Results
{''.join(search_summaries)}

## Key Insights Summary
"""
        
        # Generate insights summary using LLM
        if self.debug_mode:
            self.logger.debug("Generating insights summary from search results...")
        
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        insights_prompt = f"""Current Date and Time: {current_datetime}

Analyze these web search results and provide key insights for the research question.

Research Question: {question}

Search Results Summary:
{''.join(search_summaries[:3])}

Provide:
1. Key findings from search results
2. Important data sources mentioned
3. Relevant methodologies or approaches
4. Current trends or recent developments (as of {datetime.now().strftime('%B %Y')})
5. Gaps or areas requiring deeper investigation

Write 3-5 paragraphs of analysis."""

        insights = await self.llm_client.generate(
            insights_prompt,
            temperature=0.7,
            purpose=ModelPurpose.REASONING
        )
        
        grounding_context += insights
        grounding_context += f"\n\n## Timestamp\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if self.debug_mode:
            self.logger.debug(f"Generated grounding context: {len(grounding_context)} chars")
        
        # Stage 2: Generate detailed research plan with grounding context
        self.logger.info("📋 Generating research plan with grounding context...")
        
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plan_prompt = f"""Current Date and Time: {current_datetime}

You are a research planner. Based on the grounding context from web searches, create a detailed research plan.

Research Question: {question}

Grounding Context:
{grounding_context[:2000]}

Based on the grounding context and research question, provide:
1. Main hypothesis or research focus
2. Required data sources (be specific with tickers, datasets, APIs)
3. Analysis methods needed
4. Expected outputs and deliverables

Format as JSON with keys: hypothesis, data_sources, methods, outputs, key_insights"""
        
        if self.debug_mode:
            self.logger.debug(f"Sending planning prompt to LLM (length: {len(plan_prompt)} chars)")
            self.logger.debug(f"Current date/time: {current_datetime}")
        
        plan_text = await self.llm_client.generate(
            plan_prompt,
            temperature=0.7,
            purpose=ModelPurpose.REASONING
        )
        
        if self.debug_mode:
            self.logger.debug(f"LLM plan response (length: {len(plan_text)} chars):\n{plan_text[:500]}...")
        
        # Parse plan
        try:
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                if self.debug_mode:
                    self.logger.debug("Successfully parsed plan as JSON")
            else:
                plan = {"plan_text": plan_text}
                if self.debug_mode:
                    self.logger.debug("Could not extract JSON, using text format")
        except Exception as e:
            plan = {"plan_text": plan_text}
            if self.debug_mode:
                self.logger.debug(f"JSON parsing error: {e}")
        
        # Add grounding context and search results to plan
        plan["grounding_context"] = grounding_context
        plan["search_queries"] = search_queries
        plan["total_search_results"] = len(all_search_results)
        plan["web_search_results"] = all_search_results[:20]  # Keep top 20 results
        
        if self.debug_mode:
            self.logger.debug(f"Plan includes {len(search_queries)} queries and {len(all_search_results)} search results")
        
        return plan
    
    async def _collect_data(self, plan: Dict, project_dir: Path) -> Dict[str, Any]:
        """Collect required data based on plan with grounding context"""
        
        results = {
            "downloaded_files": [],
            "datasets": {},
            "web_context": []
        }
        
        # Extract grounding context from plan
        grounding_context = plan.get("grounding_context", "")
        grounding_summary = grounding_context[:500] if grounding_context else "No grounding context available"
        
        if self.debug_mode:
            self.logger.debug(f"Using grounding context for data collection (length: {len(grounding_context)} chars)")
        
        # Stage 1: Additional search for data sources with grounding context
        self.logger.info("🔍 Searching for specific data sources...")
        
        plan_text = json.dumps(plan)
        
        # Search for relevant data sources using grounding context
        data_search_query = f"{plan.get('hypothesis', '')} financial data sources market data"
        if self.debug_mode:
            self.logger.debug(f"Data source search query: {data_search_query}")
        
        data_search = await self.web_search.search(
            data_search_query,
            max_results=5
        )
        
        if data_search["success"]:
            results["web_context"] = data_search["results"]
            
            # Save search results with grounding context reference
            self.file_saver.base_dir = project_dir
            data_context = {
                "search_results": data_search,
                "grounding_context_summary": grounding_summary,
                "timestamp": datetime.now().isoformat()
            }
            await self.file_saver.save_json(
                data_context,
                "data_source_search.json"
            )
            if self.debug_mode:
                self.logger.debug("Saved data source search results with grounding context")
        
        # Extract tickers/datasets from plan and grounding context
        import re
        
        # Extract from plan
        tickers_from_plan = re.findall(r'\b([A-Z]{1,5})\b', plan_text)
        
        # Extract from grounding context
        tickers_from_context = re.findall(r'\b([A-Z]{2,5})\b', grounding_context[:1000])
        
        # Combine and filter
        all_tickers = list(set(tickers_from_plan + tickers_from_context))
        
        # Filter out common words that look like tickers
        excluded = {'THE', 'AND', 'FOR', 'ARE', 'THIS', 'WITH', 'FROM', 'THAT', 'WILL', 'HAVE', 'USA', 'USD', 'US'}
        tickers = [t for t in all_tickers if t not in excluded]
        
        # Common research tickers as fallback
        default_tickers = ['SPY', 'QQQ']
        tickers = list(set(tickers[:5] if tickers else default_tickers))
        
        if self.debug_mode:
            self.logger.debug(f"Extracted tickers from plan and grounding context: {tickers}")
        
        # Download financial data
        for ticker in tickers:
            self.logger.info(f"📊 Downloading data for {ticker}")
            if self.debug_mode:
                self.logger.debug(f"Calling data downloader for {ticker} from 2020-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
            
            result = await self.data_downloader.download_financial_data(
                ticker=ticker,
                start_date="2020-01-01",
                end_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            if result["success"]:
                # Copy to project directory
                import shutil
                dest_file = project_dir / f"{ticker}_data.csv"
                shutil.copy(result["cache_file"], dest_file)
                
                if self.debug_mode:
                    self.logger.debug(f"Downloaded {ticker}: {result['rows']} rows")
                    self.logger.debug(f"Date range: {result['start_date']} to {result['end_date']}")
                    self.logger.debug(f"Saved to: {dest_file}")
                
                results["downloaded_files"].append(str(dest_file))
                results["datasets"][ticker] = {
                    "file": str(dest_file),
                    "rows": result["rows"],
                    "date_range": f"{result['start_date']} to {result['end_date']}"
                }
            else:
                if self.debug_mode:
                    self.logger.debug(f"Failed to download {ticker}: {result.get('error', 'Unknown error')}")
        
        return results
    
    async def _generate_and_execute_analysis(self, 
                                            question: str,
                                            data_results: Dict,
                                            project_dir: Path,
                                            grounding_context: str = "") -> Dict[str, Any]:
        """Generate analysis code and execute it with grounding context"""
        
        # Prepare grounding context summary
        grounding_summary = grounding_context[:800] if grounding_context else "No grounding context available"
        
        # Generate analysis code
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        code_prompt = f"""Current Date and Time: {current_datetime}

Generate Python code to analyze this research question with grounding context.

Research Question: {question}

Grounding Context (Key Insights):
{grounding_summary}

Available data files:
{json.dumps(data_results.get('datasets', {}), indent=2)}

Requirements:
1. Load the data files
2. Calculate returns, volatility, and key statistics
3. Perform analysis aligned with the grounding context insights
4. Save results to 'analysis_results.json'

Use pandas, numpy, scipy. Write complete, executable code with error handling.
Save all results to files in the current directory."""
        
        self.logger.info("Generating analysis code with grounding context...")
        if self.debug_mode:
            self.logger.debug(f"Analysis code prompt length: {len(code_prompt)} chars")
            self.logger.debug(f"Prompt excerpt:\n{code_prompt[:400]}...")
            self.logger.debug(f"Grounding context included: {len(grounding_summary)} chars")
            self.logger.debug(f"Current date/time: {current_datetime}")
        
        code = await self.llm_client.generate_code(
            prompt=code_prompt,
            language="python"
        )
        
        if self.debug_mode:
            self.logger.debug(f"Generated code length: {len(code)} chars")
            self.logger.debug(f"Code preview:\n{code[:300]}...")
        
        # Extract Python code from markdown if present
        import re
        code_match = re.search(r'```python\n(.*?)\n```', code, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            if self.debug_mode:
                self.logger.debug("Extracted code from ```python``` block")
        else:
            # Try without language specifier
            code_match = re.search(r'```\n(.*?)\n```', code, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                if self.debug_mode:
                    self.logger.debug("Extracted code from ``` block")
        
        # Save generated code
        code_file = project_dir / "analysis_code.py"
        with open(code_file, 'w') as f:
            f.write(code)
        
        if self.debug_mode:
            self.logger.debug(f"Saved analysis code to: {code_file}")
        
        # Execute code
        self.logger.info("Executing analysis code...")
        self.code_executor.working_dir = str(project_dir)
        
        if self.debug_mode:
            self.logger.debug(f"Code executor working dir: {self.code_executor.working_dir}")
            self.logger.debug("Starting code execution with 180s timeout")
        
        exec_result = await self.code_executor.execute_code(
            code=code,
            timeout=180,
            save_artifacts=True
        )
        
        if self.debug_mode:
            self.logger.debug(f"Code execution completed. Success: {exec_result['success']}")
            self.logger.debug(f"Stdout length: {len(exec_result['stdout'])} chars")
            self.logger.debug(f"Stderr length: {len(exec_result.get('stderr', ''))} chars")
            if exec_result['stdout']:
                self.logger.debug(f"Full stdout:\n{exec_result['stdout']}")
            if exec_result.get('stderr'):
                self.logger.debug(f"Full stderr:\n{exec_result['stderr']}")
        
        results = {
            "code_file": str(code_file),
            "execution": {
                "success": exec_result["success"],
                "stdout": exec_result["stdout"][-1000:] if len(exec_result["stdout"]) > 1000 else exec_result["stdout"],
                "stderr": exec_result["stderr"][-500:] if len(exec_result["stderr"]) > 500 else exec_result["stderr"]
            }
        }
        
        # Try to load analysis results if saved
        results_file = project_dir / "analysis_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results["analysis_output"] = json.load(f)
        
        return results
    
    async def _generate_visualizations(self,
                                      analysis_results: Dict,
                                      project_dir: Path,
                                      grounding_context: str = "") -> Dict[str, Any]:
        """Generate visualizations from analysis using LLM code generation with grounding context"""
        
        viz_results = {"plots": []}
        
        # Prepare grounding context summary
        grounding_summary = grounding_context[:600] if grounding_context else "No grounding context available"
        
        # First, probe the file structure to understand available data
        self.logger.info("🔍 Probing data files for visualization context...")
        
        if self.debug_mode:
            self.logger.debug(f"Project directory for viz: {project_dir}")
            self.logger.debug("Starting data structure probe...")
            self.logger.debug(f"Grounding context available: {len(grounding_context)} chars")
        
        probe_code = '''
import pandas as pd
import glob
import json

result = {"files": [], "file_details": {}}

csv_files = glob.glob("*_data.csv")
for csv_file in csv_files:
    try:
        # Quick peek at structure
        df = pd.read_csv(csv_file, nrows=5)
        
        # Check if ticker metadata row exists
        if df.iloc[0].astype(str).str.contains('Ticker', case=False).any():
            df = pd.read_csv(csv_file, skiprows=[1], nrows=100)
        
        # Parse dates
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        ticker = csv_file.replace('_data.csv', '')
        result["files"].append(ticker)
        result["file_details"][ticker] = {
            "columns": list(df.columns),
            "rows": len(pd.read_csv(csv_file)),
            "date_range": f"{df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else "N/A",
            "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
        }
    except Exception as e:
        result["file_details"][csv_file] = {"error": str(e)}

print(json.dumps(result, indent=2, default=str))
'''
        
        self.code_executor.working_dir = str(project_dir)
        probe_result = await self.code_executor.execute_code(probe_code, timeout=30)
        
        if self.debug_mode:
            self.logger.debug(f"Probe completed. Success: {probe_result['success']}")
            if probe_result['success']:
                self.logger.debug(f"Probe output:\n{probe_result['stdout'][:800]}...")
            else:
                self.logger.debug(f"Probe error: {probe_result.get('stderr', 'No error output')}")
        
        file_structure = {}
        if probe_result["success"] and probe_result["stdout"]:
            try:
                import json
                file_structure = json.loads(probe_result["stdout"])
                
                if self.debug_mode:
                    self.logger.debug(f"Parsed file structure: {len(file_structure.get('files', []))} files")
                    self.logger.debug(f"File list: {file_structure.get('files', [])}")
            except:
                file_structure = {"raw_output": probe_result["stdout"][:500]}
                
                if self.debug_mode:
                    self.logger.debug("Failed to parse probe output as JSON")
        
        # Get analysis context
        analysis_code = ""
        if "code_file" in analysis_results:
            try:
                with open(analysis_results["code_file"], 'r') as f:
                    analysis_code = f.read()[:1000]  # First 1000 chars
                
                if self.debug_mode:
                    self.logger.debug(f"Loaded analysis code: {len(analysis_code)} chars")
            except:
                if self.debug_mode:
                    self.logger.debug("Failed to load analysis code file")
                pass
        
        # Generate visualization code with LLM using full context
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        viz_prompt = f"""Current Date and Time: {current_datetime}

Generate Python code for creating comprehensive visualizations of financial data.

GROUNDING CONTEXT:
{grounding_summary}

ANALYSIS CONTEXT:
- Analysis Results: {analysis_results.get('execution', {}).get('output', 'N/A')[:500]}
- Available Data Files: {file_structure.get('files', [])}
- File Details: {json.dumps(file_structure.get('file_details', {}), indent=2, default=str)[:1000]}

ANALYSIS CODE SNIPPET:
{analysis_code[:800] if analysis_code else 'N/A'}

REQUIREMENTS:
1. Create 3 visualizations PER TICKER that align with the grounding context insights:
   - Price time series with proper date formatting (YYYY-MM-DD on X-axis)
   - Returns distribution histogram
   - Rolling 20-day volatility with date formatting

2. DATA LOADING (CRITICAL):
   - Price time series with proper date formatting (YYYY-MM-DD on X-axis)
   - Returns distribution histogram
   - Rolling 20-day volatility with date formatting

2. DATA LOADING (CRITICAL):
   - Files are named: {{ticker}}_data.csv
   - CSVs have a "Ticker" metadata row - MUST skip with skiprows=[1]
   - Columns: Date, Open, High, Low, Close, Volume
   - Date must be parsed and set as index
   - All price columns must be converted to numeric with pd.to_numeric()

3. DATE FORMATTING (REQUIRED):
   - Import: matplotlib.dates as mdates
   - Use: ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
   - Use: ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
   - Rotate labels: plt.xticks(rotation=45, ha='right')

4. CODE STRUCTURE:
```python
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import glob

sns.set_style("whitegrid")

csv_files = glob.glob("*_data.csv")
for csv_file in csv_files:
    try:
        # Load with ticker row skipped
        df = pd.read_csv(csv_file, skiprows=[1])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Convert to numeric
        for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        base_name = csv_file.replace('_data.csv', '') + '_data'
        
        # Plot 1: Prices with date formatting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], linewidth=1.5)
        ax.set_title(f'{{base_name}} - Closing Prices')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{{base_name}}_prices.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Returns distribution
        # ... (similar structure)
        
        # Plot 3: Volatility with date formatting
        # ... (similar structure)
        
        print(f"✓ Created 3 plots for {{csv_file}}")
    except Exception as e:
        print(f"✗ Error: {{e}}")
        import traceback
        traceback.print_exc()

print("\\n✅ Visualization complete!")
```

Generate complete, working code following this pattern for all tickers in the data."""

        self.logger.info("🎨 Generating visualization code with LLM...")
        
        if self.debug_mode:
            self.logger.debug(f"Visualization prompt length: {len(viz_prompt)} chars")
            self.logger.debug(f"Prompt excerpt:\n{viz_prompt[:600]}...")
            self.logger.debug(f"Current date/time: {current_datetime}")
        
        viz_code = await self.llm_client.generate_code(
            viz_prompt,
            language="python"
        )
        
        if self.debug_mode:
            self.logger.debug(f"Generated viz code length: {len(viz_code)} chars")
            self.logger.debug(f"Code preview:\n{viz_code[:500]}...")
        
        # Extract code from markdown
        import re
        if "```python" in viz_code:
            match = re.search(r'```python\n(.*?)\n```', viz_code, re.DOTALL)
            if match:
                viz_code = match.group(1)
                if self.debug_mode:
                    self.logger.debug("Extracted viz code from ```python``` block")
        elif "```" in viz_code:
            match = re.search(r'```\n(.*?)\n```', viz_code, re.DOTALL)
            if match:
                viz_code = match.group(1)
                if self.debug_mode:
                    self.logger.debug("Extracted viz code from ``` block")
        
        # Execute visualization code dynamically (no file saved)
        self.logger.info("📊 Executing generated visualizations dynamically...")
        self.code_executor.working_dir = str(project_dir)
        
        if self.debug_mode:
            self.logger.debug("Starting dynamic visualization execution with 120s timeout")
            self.logger.debug(f"Visualization code length: {len(viz_code)} chars")
        
        exec_result = await self.code_executor.execute_code(
            code=viz_code,
            timeout=120
        )
        
        if self.debug_mode:
            self.logger.debug(f"Viz execution completed. Success: {exec_result['success']}")
            self.logger.debug(f"Stdout: {exec_result['stdout'] if exec_result['stdout'] else 'No stdout'}")
            if exec_result.get('stderr'):
                self.logger.debug(f"Stderr: {exec_result['stderr']}")
        
        # No code file saved - executed dynamically
        viz_results["code_file"] = None
        viz_results["execution"] = {
            "success": exec_result["success"],
            "output": exec_result["stdout"][-500:] if exec_result["stdout"] else "",
            "error": exec_result["stderr"][-500:] if exec_result.get("stderr") else ""
        }
        
        # List generated plot files
        plot_files = list(project_dir.glob("*.png"))
        viz_results["plots"] = [str(f.name) for f in plot_files]
        
        if self.debug_mode:
            self.logger.debug(f"Found {len(plot_files)} PNG files in project directory")
            if plot_files:
                self.logger.debug(f"Plot files: {[f.name for f in plot_files]}")
        
        if not viz_results["plots"]:
            self.logger.warning("No plots were generated. Check execution errors.")
            
            if self.debug_mode:
                self.logger.debug("No PNG files found after visualization execution")
        else:
            self.logger.info(f"✅ Generated {len(viz_results['plots'])} visualizations")
        
        return viz_results
    
    async def _generate_report(self,
                              question: str,
                              artifacts: Dict,
                              project_dir: Path,
                              grounding_context: str = "") -> Dict[str, Any]:
        """Generate research report with grounding context"""
        
        # Prepare grounding context summary
        grounding_summary = grounding_context[:1000] if grounding_context else "No grounding context available"
        
        if self.debug_mode:
            self.logger.debug("Generating research report with grounding context...")
            self.logger.debug(f"Number of datasets: {len(artifacts['stages']['data_collection'].get('datasets', {}))}")
            self.logger.debug(f"Analysis success: {artifacts['stages']['analysis']['execution']['success']}")
            self.logger.debug(f"Number of plots: {len(artifacts['stages']['visualizations'].get('plots', []))}")
            self.logger.debug(f"Grounding context: {len(grounding_context)} chars")
        
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_prompt = f"""Current Date and Time: {current_datetime}

Generate a comprehensive research report grounded in the web search context.

Research Question: {question}

Grounding Context (Web Search Insights):
{grounding_summary}

Project Summary:
- Data collected: {len(artifacts['stages']['data_collection'].get('datasets', {}))} datasets
- Analysis executed: {'Yes' if artifacts['stages']['analysis']['execution']['success'] else 'No'}
- Visualizations: {len(artifacts['stages']['visualizations'].get('plots', []))} plots created

Write a professional research report with:
1. Executive Summary (referencing grounding context insights)
2. Literature Review (based on web search findings)
3. Methodology
4. Results and Analysis
5. Discussion (connecting findings to grounding context)
6. Conclusions
7. Limitations and Future Research

Format in Markdown with proper sections and citations to sources from grounding context."""
        
        if self.debug_mode:
            self.logger.debug(f"Report prompt length: {len(report_prompt)} chars")
            self.logger.debug(f"Prompt excerpt:\n{report_prompt[:500]}...")
            self.logger.debug(f"Current date/time: {current_datetime}")
        
        report = await self.llm_client.write_content(
            prompt=report_prompt,
            content_type="academic"
        )
        
        if self.debug_mode:
            self.logger.debug(f"Generated report length: {len(report)} chars")
            self.logger.debug(f"Report preview:\n{report[:400]}...")
        
        # Save report
        report_file = project_dir / "research_report.md"
        with open(report_file, 'w') as f:
            f.write(f"# Research Report\n\n")
            f.write(f"**Question:** {question}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
            f.write("---\n\n")
            f.write(report)
        
        if self.debug_mode:
            self.logger.debug(f"Saved report to: {report_file}")
        
        return {
            "file": str(report_file),
            "preview": report[:500] + "..." if len(report) > 500 else report
        }
    
    async def _generate_section_visualizations(self,
                                               question: str,
                                               report: Dict,
                                               data_results: Dict,
                                               project_dir: Path,
                                               grounding_context: str = "") -> Dict[str, Any]:
        """Generate comprehensive visualizations for each report section using LLM with grounding context"""
        
        # Prepare grounding context summary
        grounding_summary = grounding_context[:800] if grounding_context else "No grounding context available"
        
        self.logger.info("📊 Analyzing report and generating section-specific visualizations with grounding context...")
        
        if self.debug_mode:
            self.logger.debug(f"Section viz project dir: {project_dir}")
            self.logger.debug(f"Grounding context: {len(grounding_context)} chars")
        
        # Read the generated report
        report_file = project_dir / "research_report.md"
        with open(report_file, 'r') as f:
            report_content = f.read()
        
        if self.debug_mode:
            self.logger.debug(f"Read report: {len(report_content)} chars")
        
        # Parse report sections
        import re
        section_titles = re.findall(r'^#{1,3}\s+(.+)$', report_content, flags=re.MULTILINE)
        
        if self.debug_mode:
            self.logger.debug(f"Found {len(section_titles)} section titles")
            self.logger.debug(f"Section titles: {section_titles[:10]}")
        
        # Probe data structure again
        if self.debug_mode:
            self.logger.debug("Probing data structure for section visualizations...")
        
        probe_code = '''
import pandas as pd
import glob
import json

result = {"tickers": [], "structure": {}}
csv_files = glob.glob("*_data.csv")

for csv_file in csv_files:
    ticker = csv_file.replace('_data.csv', '')
    df = pd.read_csv(csv_file, skiprows=[1], nrows=10)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    result["tickers"].append(ticker)
    result["structure"][ticker] = {
        "columns": list(df.columns),
        "sample": df.head(2).to_dict('records')
    }

print(json.dumps(result, indent=2, default=str))
'''
        
        self.code_executor.working_dir = str(project_dir)
        probe_result = await self.code_executor.execute_code(probe_code, timeout=30)
        
        if self.debug_mode:
            self.logger.debug(f"Data probe completed. Success: {probe_result['success']}")
            if probe_result['success']:
                self.logger.debug(f"Probe output:\n{probe_result['stdout'][:600]}...")
        
        data_structure = {}
        if probe_result["success"] and probe_result["stdout"]:
            try:
                import json
                data_structure = json.loads(probe_result["stdout"])
                
                if self.debug_mode:
                    self.logger.debug(f"Parsed data structure: {len(data_structure.get('tickers', []))} tickers")
            except:
                data_structure = {"raw": probe_result["stdout"][:300]}
                
                if self.debug_mode:
                    self.logger.debug("Failed to parse data structure as JSON")
        
        # Generate comprehensive visualization code with full context
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        section_viz_prompt = f"""Current Date and Time: {current_datetime}

Generate Python code for creating comprehensive section-specific visualizations for a research report.

GROUNDING CONTEXT (Web Search Insights):
{grounding_summary}

RESEARCH CONTEXT:
Question: {question}

Report Sections:
{chr(10).join([f"  {i+1}. {title}" for i, title in enumerate(section_titles[:12])])}

Report Excerpt (first 1500 chars):
{report_content[:1500]}

Data Available:
- Tickers: {data_structure.get('tickers', [])}
- Structure: {json.dumps(data_structure.get('structure', {}), indent=2, default=str)[:600]}

TASK:
Create 7 publication-quality visualizations that align with BOTH the report sections AND the grounding context insights:

1. **section_1_performance.png**: Normalized price comparison (all tickers start at 100)
2. **section_2_returns.png**: Returns distribution comparison (histograms side-by-side)
3. **section_3_volatility.png**: Rolling 30-day volatility (annualized) comparison
4. **section_4_volume.png**: Trading volume analysis with moving averages
5. **section_5_correlation.png**: Returns correlation heatmap (if multiple tickers)
6. **section_6_risk_return.png**: Risk-return scatter plot (volatility vs return)
7. **section_7_cumulative.png**: Cumulative returns over time

CRITICAL REQUIREMENTS:
1. **Data Loading**:
```python
csv_files = glob.glob("*_data.csv")
datasets = {{}}
for csv_file in csv_files:
    df = pd.read_csv(csv_file, skiprows=[1])  # Skip ticker row
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    ticker = csv_file.replace('_data.csv', '')
    datasets[ticker] = df
```

2. **Date Formatting** (REQUIRED for all time-series plots):
```python
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right')
```

3. **Professional Styling**:
- Use seaborn style: plt.style.use('seaborn-v0_8-darkgrid')
- Figure sizes: (14, 7) for time series, (10, 8) for heatmaps
- DPI: 150
- linewidth: 2.5, alpha: 0.8 for lines
- Include titles, labels, legends, grids

4. **Error Handling**:
- Wrap each visualization in try/except
- Print success/error messages for each

5. **Calculations**:
- Returns: df['Close'].pct_change(fill_method=None)
- Annualized volatility: returns.rolling(30).std() * np.sqrt(252)
- Normalized prices: (prices / prices.iloc[0]) * 100

Generate COMPLETE, WORKING Python code that creates all 7 visualizations."""

        if self.debug_mode:
            self.logger.debug(f"Section viz prompt length: {len(section_viz_prompt)} chars")
            self.logger.debug(f"Prompt excerpt:\n{section_viz_prompt[:600]}...")
            self.logger.debug(f"Current date/time: {current_datetime}")
        
        self.logger.info("🎨 Generating section visualization code with LLM...")
        viz_code = await self.llm_client.generate_code(
            section_viz_prompt,
            language="python"
        )
        
        if self.debug_mode:
            self.logger.debug(f"Generated section viz code length: {len(viz_code)} chars")
            self.logger.debug(f"Code preview:\n{viz_code[:500]}...")
        
        # Extract code from markdown
        if "```python" in viz_code:
            match = re.search(r'```python\n(.*?)\n```', viz_code, re.DOTALL)
            if match:
                viz_code = match.group(1)
                if self.debug_mode:
                    self.logger.debug("Extracted section viz code from ```python``` block")
        elif "```" in viz_code:
            match = re.search(r'```\n(.*?)\n```', viz_code, re.DOTALL)
            if match:
                viz_code = match.group(1)
                if self.debug_mode:
                    self.logger.debug("Extracted section viz code from ``` block")
        
        # Execute section visualization code dynamically (no file saved)
        self.logger.info("🎨 Executing section visualizations dynamically...")
        self.code_executor.working_dir = str(project_dir)
        
        if self.debug_mode:
            self.logger.debug("Starting dynamic section viz execution with 180s timeout")
            self.logger.debug(f"Section visualization code length: {len(viz_code)} chars")
        
        exec_result = await self.code_executor.execute_code(
            code=viz_code,
            timeout=180
        )
        
        if self.debug_mode:
            self.logger.debug(f"Section viz execution completed. Success: {exec_result['success']}")
            self.logger.debug(f"Stdout: {exec_result['stdout'] if exec_result['stdout'] else 'No stdout'}")
            if exec_result.get('stderr'):
                self.logger.debug(f"Stderr: {exec_result['stderr']}")
        
        # Collect results (no code file saved - executed dynamically)
        section_viz_results = {
            "code_file": None,
            "execution": {
                "success": exec_result["success"],
                "output": exec_result["stdout"][-1000:] if exec_result["stdout"] else "",
                "error": exec_result["stderr"][-1000:] if exec_result.get("stderr") else ""
            }
        }
        
        # List newly generated section plot files
        all_plots = list(project_dir.glob("section_*.png"))
        section_viz_results["plots"] = [str(f.name) for f in all_plots]
        
        if self.debug_mode:
            self.logger.debug(f"Found {len(all_plots)} section_*.png files")
            if all_plots:
                self.logger.debug(f"Section plot files: {[f.name for f in all_plots]}")
        
        if section_viz_results["plots"]:
            self.logger.info(f"✅ Generated {len(section_viz_results['plots'])} section-specific visualizations")
        else:
            self.logger.warning("⚠️  No section visualizations generated, check execution output")
            
            if self.debug_mode:
                self.logger.debug("No section_*.png files found after execution")
        
        return section_viz_results

