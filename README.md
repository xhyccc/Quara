# QuARA - Quantitative Academic Research Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QuARA** is an AI-powered multi-agent research assistant that automates the entire quantitative research workflow—from question formulation to report generation. Built with LLM-driven automation, QuARA downloads financial data, generates analysis code, creates visualizations, and produces comprehensive research reports grounded in real-time web search context.

## 🌟 Key Features

### 🤖 Fully Automated Research
- **One-Command Research**: Ask a question, get a complete research project
- **Intelligent Data Collection**: Automatically identifies and downloads relevant financial datasets
- **Dynamic Code Generation**: LLM generates and executes Python analysis code on-the-fly
- **Smart Visualizations**: Creates publication-quality plots without saving intermediate code files
- **Web-Grounded Reports**: Every research stage informed by real-time web search insights

### 📊 Comprehensive Analysis Pipeline
1. **Research Planning** - Web search + LLM-driven research design
2. **Data Collection** - Automatic financial data download (e.g., yfinance)
3. **Statistical Analysis** - Generated Python code for returns, volatility, correlations
4. **Visualizations** - Time series, distributions, heatmaps, risk-return plots
5. **Report Generation** - Academic-style markdown reports with literature review
6. **Section-Specific Viz** - Targeted visualizations aligned with report sections

### 🎯 Debug Mode
- Comprehensive debug logging with `--debug` flag
- 60+ debug checkpoints across all stages
- Full visibility into LLM prompts, responses, and execution flows

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/xhyccc/Quara.git
cd Quara

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r quara/requirements.txt

# Configure API key
cp quara/.env.example quara/.env
# Edit .env and add your SILICONFLOW_API_KEY
```

### Basic Usage

```bash
# Interactive mode
python quara/cli.py

# Automated research (recommended!)
python quara/cli.py
> auto-research Why did Apple stock drop today?

# With debug logging
python quara/cli.py --debug
> auto-research Analyze the correlation between US-China trade tensions and SPY returns
```

### Example Commands

```bash
# Research & Analysis
auto-research <question>     # 🔥 Full automated workflow
ask <question>               # Quick LLM Q&A
analyze <topic>              # Multi-stage deep analysis

# Code & Writing
code <task>                  # Generate Python code
write <topic>                # Academic writing

# System
config                       # Show configuration
test                         # Test LLM connection
help                         # Show all commands
```

## 📁 Project Structure

```
Quara/
├── quara/                          # Main package
│   ├── cli.py                      # Interactive CLI
│   ├── quara/
│   │   ├── core/
│   │   │   ├── automated_executor.py  # Research automation engine
│   │   │   ├── system.py              # Multi-agent orchestration
│   │   │   └── base.py                # Base classes
│   │   ├── agents/                    # Research agents
│   │   ├── tools/                     # Data & code tools
│   │   └── utils/                     # LLM clients & utilities
│   └── requirements.txt
├── research_projects/              # Generated research outputs (gitignored)
├── data_cache/                     # Downloaded data cache (gitignored)
├── .env                            # API keys (gitignored)
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Required
SILICONFLOW_API_KEY=your_api_key_here

# Optional (defaults shown)
QUARA_API_BASE=https://api.siliconflow.cn/v1
REASONING_MODEL=deepseek-ai/DeepSeek-V3.2-Exp
CODE_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
WRITING_MODEL=deepseek-ai/DeepSeek-V3.2-Exp
QUARA_TIMEOUT=180
QUARA_PROJECT_ROOT=./research_projects
```

### Model Configuration

QuARA uses specialized models for different tasks:
- **Reasoning** (DeepSeek-V3.2): Agent thinking, planning, analysis
- **Code** (Qwen-Coder-32B): Python/R/SQL generation
- **Writing** (DeepSeek-V3.2): Academic reports, documentation

## 📖 Example Workflow

```bash
python quara/cli.py --debug
```

```
QuARA> auto-research Why does Microsoft stock drop after earnings?
```

**What happens:**
1. 🔍 **Grounding Search**: Executes 5-7 web searches, saves context to `grounding_context.md`
2. 📋 **Plan Generation**: LLM creates research plan identifying data needs
3. 📊 **Data Download**: Auto-downloads MSFT, sector ETFs, comparison tickers
4. 💻 **Analysis Code**: Generates Python code for returns, volatility, statistical tests
5. 📈 **Visualizations**: Creates time series, distributions, correlation heatmaps
6. 📝 **Report**: Writes academic-style report with executive summary, methodology, findings
7. 🎨 **Section Viz**: Generates 7 publication-quality plots aligned with report sections

**Output** (in `research_projects/research_YYYYMMDD_HHMMSS/`):
- `grounding_context.md` - Web search insights
- `MSFT_data.csv`, `SPY_data.csv`, etc. - Downloaded datasets
- `analysis_code.py` - Generated analysis code
- `*.png` - Visualization plots
- `research_report.md` - Final report

## 🛠️ Advanced Features

### Debug Logging

Enable comprehensive debug output:

```bash
python quara/cli.py --debug
```

Debug mode shows:
- LLM prompt lengths and excerpts
- Generated code previews
- Execution timings and success status
- Data structure probes
- File operations and paths

See [DEBUG_LOGGING.md](DEBUG_LOGGING.md) for details.

### Dynamic Code Execution

QuARA executes visualization code **dynamically in memory** without saving intermediate files:
- Cleaner project directories
- Faster execution
- Focus on actual outputs (PNG plots)

### Web-Grounded Research

Every research stage includes real-time web search context:
- Multi-query search strategy (5-7 queries per research)
- LLM-generated search queries
- Comprehensive context markdown saved for reproducibility

## 📚 Documentation

- [CLI Guide](CLI_GUIDE.md) - Complete CLI reference
- [Quick Reference](CLI_QUICK_REFERENCE.md) - Command cheat sheet
- [Automated Research Guide](AUTOMATED_RESEARCH_GUIDE.md) - Deep dive into automation
- [Debug Logging](DEBUG_LOGGING.md) - Debug mode documentation
- [LLM Configuration](quara/LLM_CONFIGURATION.md) - Model setup
- [Quickstart](quara/QUICKSTART.md) - Getting started

## 🔬 Research Examples

### Financial Markets
```bash
auto-research Analyze the correlation between VIX and SPY returns during market downturns
auto-research Why did tech stocks outperform in Q4 2024?
auto-research Impact of Federal Reserve rate decisions on bond yields
```

### Comparative Analysis
```bash
auto-research Compare volatility patterns of AAPL vs MSFT over the past 5 years
auto-research Which sector performed best during the recent earnings season?
```

### Event Studies
```bash
auto-research How did NVDA stock react to AI chip export restrictions?
auto-research Market impact of geopolitical tensions on energy stocks
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [SiliconFlow API](https://siliconflow.cn/) (DeepSeek, Qwen models)
- Financial data via [yfinance](https://github.com/ranaroussi/yfinance)
- Inspired by modern AI agent architectures

## 📧 Contact

For questions or feedback, please open an issue on GitHub. Contact me via haoyi.xiong.fr@ieee.org.

---

**Made with ❤️ for quantitative research automation**
