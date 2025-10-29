# QuARA (Quantitative Academic Research Agent)

A sophisticated multi-agent system for autonomous scientific discovery and analysis, implementing the complete research workflow from hypothesis generation to academic publication.

## Overview

QuARA is a multi-agent framework that automates the entire quantitative research lifecycle, from initial research questions to complete academic manuscripts. Built on ReAct (Reasoning and Acting) agents with a centralized Master Control Protocol (MCP) for secure communication and coordination.

## System Architecture

### Multi-Agent Team

- **🎯 Orchestrator Agent**: Principal investigator managing the research workflow
- **🧠 Theorist Agent**: Problem definition, literature synthesis, and hypothesis generation
- **📚 Librarian Agent**: Data collection and domain knowledge curation
- **⚗️ Methodologist Agent**: Experimental design and statistical planning
- **📊 Analyst Agent**: Quantitative execution and analysis
- **✍️ Scribe Agent**: Academic writing and manuscript generation

### Core Components

- **Master Control Protocol (MCP)**: Secure message-passing hub for agent coordination
- **Zettelkasten Memory**: Long-term knowledge storage with vector similarity search
- **Human-in-the-Loop (HITL)**: Strategic validation checkpoints
- **Reproducibility Framework**: Complete audit trail and artifact generation

## Features

### 🔬 Complete Research Automation
- End-to-end workflow from research question to publication
- Rigorous experimental design with causal inference
- Automated literature review and gap analysis
- Statistical analysis with baseline and benchmark comparisons

### 🧪 Scientific Rigor
- Causal vs. predictive task classification
- DoWhy-based causal inference with robustness checks
- Formal hypothesis testing and effect size estimation
- Publication-quality visualizations

### 🔒 Security & Validation
- Agent-tool access control through MCP security policies
- Human-in-the-Loop validation at critical decision points
- Sandboxed code execution for statistical analysis
- Complete reproducibility with Docker containers

### 🧠 Intelligent Memory
- Zettelkasten-based knowledge graph construction
- Mem0-style memory updates (ADD, UPDATE, DELETE, NOOP)
- Vector similarity search for knowledge retrieval
- Project-specific memory organization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd quara

# Install dependencies
pip install -r requirements.txt

# Configure LLM (Required for full functionality)
# Copy the example configuration
cp .env.example .env

# Edit .env and add your API key
# For SiliconFlow + DeepSeek-V3 (recommended):
#   SILICONFLOW_API_KEY=your_api_key_here
#   QUARA_LLM_MODEL=deepseek-ai/DeepSeek-V3

# Or set environment variable
export SILICONFLOW_API_KEY="your_api_key_here"
```

See [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md) for detailed configuration options.

## Quick Start

### Configure LLM Provider

QuARA supports multiple LLM providers. **SiliconFlow with DeepSeek-V3** is recommended:

```bash
# Get API key from https://siliconflow.cn
export SILICONFLOW_API_KEY="your_api_key"
export QUARA_LLM_MODEL="deepseek-ai/DeepSeek-V3"
```

### Simple Usage

```python
import asyncio
from quara import conduct_research

async def main():
    result = await conduct_research(
        "Analyze the effect of screen time on adolescent sleep quality"
    )
    
    if result['success']:
        print(f"Research completed! Project ID: {result['project_id']}")
    else:
        print(f"Research failed: {result['error']}")

asyncio.run(main())
```

### Advanced Usage with LLM Configuration

```python
import asyncio
from quara import QuARASystem

async def main():
    # Initialize system with LLM configuration
    llm_config = {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3",
        "api_key": "your_api_key",
        "temperature": 0.7,
        "max_tokens": 4096
    }
    
    system = QuARASystem(
        llm_config=llm_config,
        enable_memory=True,
        enable_real_tools=False,  # Use mock tools for demo
        project_root="./my_research_projects"
    )
    
    try:
        # Conduct research with detailed context
        result = await system.conduct_research(
            research_request="Study causal effects of social media on mental health",
            context={
                "user_preferences": {
                    "methodology": "causal_inference",
                    "statistical_significance": 0.05,
                    "output_format": "academic_paper"
                }
            }
        )
        
        if result['success']:
            # Get project details
            project_status = await system.get_project_status(result['project_id'])
            print(f"Generated {len(project_status['artifacts'])} artifacts")
            
            # Query system memory
            insights = await system.query_system_memory(
                "causal effect hypothesis", 
                project_id=result['project_id']
            )
            print(f"Found {len(insights)} relevant memory nodes")
        
    finally:
        await system.stop_system()

asyncio.run(main())
```

## LLM Configuration

QuARA requires an LLM for agent reasoning. Multiple providers are supported:

### SiliconFlow + DeepSeek-V3 (Recommended)

```python
from quara import QuARASystem

llm_config = {
    "provider": "siliconflow",
    "model": "deepseek-ai/DeepSeek-V3",
    "api_key": "your_siliconflow_api_key"
}

async with QuARASystem(llm_config=llm_config) as system:
    result = await system.conduct_research("Your research question")
```

### Using Environment Variables

```bash
export SILICONFLOW_API_KEY="your_api_key"
export QUARA_LLM_MODEL="deepseek-ai/DeepSeek-V3"
```

```python
# System automatically reads from environment
async with QuARASystem() as system:
    result = await system.conduct_research("Your research question")
```

### Other Providers

```python
# OpenAI
llm_config = {"provider": "openai", "model": "gpt-4", "api_key": "..."}

# Custom OpenAI-compatible endpoint
llm_config = {
    "provider": "custom",
    "model": "model-name", 
    "api_key": "...",
    "api_base": "https://your-endpoint.com/v1"
}
```

**📖 See [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md) for comprehensive configuration guide**

## Complete Example

Run the complete research workflow demonstration:

```bash
# Set API key
export SILICONFLOW_API_KEY="your_api_key"

# Run example
cd examples
python complete_research_example.py

# Or try the configuration examples
python siliconflow_config_example.py
```

This will demonstrate:
- Multi-agent coordination through MCP
- Full research workflow execution
- Memory system with knowledge storage
- HITL validation checkpoints
- Reproducibility artifact generation

## Research Workflow

### Phase 0: Iterative Design
- User provides initial research idea
- System refines through iterative feedback
- Final proposal validation and approval

### Phase 1: Theorist - Problem Definition
- Literature synthesis using RAG techniques
- Gap identification through agentic debate
- Testable hypothesis generation

### Phase 2: Librarian - Data Collection
- Academic literature retrieval (PubMed, ArXiv)
- Dataset discovery (Kaggle, UCI Repository)
- Domain knowledge synthesis

### Phase 3: Methodologist - Experimental Design
- Hypothesis formalization using OBI ontology
- Causal vs. predictive classification
- Statistical plan with evaluation framework

### Phase 4: Analyst - Quantitative Execution
- Purpose-driven data cleaning (AutoDCWorkflow)
- Statistical analysis in secure sandbox
- Causal inference with DoWhy framework
- Publication-quality visualizations

### Phase 5: Scribe - Academic Writing
- Multi-agent writing team approach
- Grounded generation from actual results
- Complete manuscript in LaTeX format

## Human-in-the-Loop Checkpoints

The system includes strategic validation points:

| Phase | Checkpoint | Validation |
|-------|------------|------------|
| Design | Final Design Approval | Research plan validation |
| Data | Data Source Validation | Dataset appropriateness |
| Methods | Methodological Validation | Statistical plan approval |
| Analysis | Results Validation | Finding interpretation |
| Writing | Manuscript Approval | Final publication review |

## Reproducibility

Every research project generates:

1. **Digital Lab Notebook**: Complete agent reasoning trace
2. **Reproducible Environment**: Docker container with exact dependencies  
3. **Data & Code Bundle**: Raw data and analysis scripts
4. **Publication Manuscript**: Camera-ready academic paper

## Architecture Details

### Master Control Protocol (MCP)

The MCP hub implements a secure publish/subscribe model:

```python
# Agents communicate through standardized interfaces
task = Task(
    originator="orchestrator", 
    target_agent="theorist",
    goal="Generate hypothesis from literature gaps"
)

# Tools are accessed through security broker
tool_request = ToolRequest(
    originator="librarian",
    tool_name="pubmed_api", 
    parameters={"query": "sleep quality adolescents"}
)
```

### Memory System

Zettelkasten-based knowledge management:

```python
# Store atomic knowledge nodes
node_id = await memory.store_knowledge(
    content="Screen time negatively affects sleep quality",
    node_type="hypothesis",
    project_id="research_123"
)

# Query with semantic similarity
results = await memory.query_memory(
    "sleep quality research findings",
    project_id="research_123"
)
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
python -m pytest tests/ -v
```

## Configuration

### Environment Variables

- `QUARA_LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING)
- `QUARA_PROJECT_ROOT`: Default project directory
- `QUARA_MEMORY_BACKEND`: Memory backend (chromadb, mock)

### Agent Configuration

Customize agent behavior through the system initialization:

```python
system = QuARASystem(
    llm_client=your_llm_client,  # Custom LLM integration
    enable_memory=True,          # Enable Zettelkasten memory
    enable_real_tools=True,      # Use real APIs vs mocks
    project_root="./projects"    # Project storage location
)
```

## Integration

### LLM Integration

QuARA supports custom LLM clients:

```python
from your_llm_library import LLMClient

llm_client = LLMClient(
    model="gpt-4",
    api_key="your-key"
)

system = QuARASystem(llm_client=llm_client)
```

### Tool Integration

Add custom tools through the MCP hub:

```python
from quara.core.base import StandardizedToolInterface

class CustomTool(StandardizedToolInterface):
    async def execute(self, parameters):
        # Your tool implementation
        return {"result": "custom_output"}
    
    def get_schema(self):
        return {"name": "custom_tool", "parameters": {...}}

# Register with system
system.mcp_hub.register_tool("custom_tool", CustomTool())
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use QuARA in your research, please cite:

```bibtex
@software{quara2024,
  title={QuARA: Quantitative Academic Research Agent},
  author={QuARA Development Team},
  year={2024},
  url={https://github.com/your-org/quara}
}
```

## Acknowledgments

- Built on principles from the design specification for autonomous scientific discovery
- Incorporates advances in multi-agent systems and causal inference
- Inspired by the ReAct framework for reasoning and acting agents

---

**QuARA**: Transforming the practice of science from human-as-technician to human-as-strategist.