#!/usr/bin/env python3
"""
QuARA CLI - Interactive Command-Line Interface
================================================

A user-friendly CLI for interacting with the QuARA multi-agent research system.

Usage:
    python cli.py                    # Interactive mode
    python cli.py research "question"  # Direct research
    python cli.py chat               # Chat with LLM
    python cli.py config             # Show configuration
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    # Try multiple locations
    env_paths = [
        Path(__file__).parent / '.env',  # quara/.env
        Path(__file__).parent.parent / '.env',  # Scientifique/.env
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

from quara import QuARASystem
from quara.utils import (
    create_llm_client,
    ModelPurpose,
    validate_research_request,
    estimate_research_time
)
from quara.core.automated_executor import AutomatedResearchExecutor


class QuARACLI:
    """Interactive CLI for QuARA system"""
    
    def __init__(self):
        self.system = None
        self.llm_client = None
        self.auto_executor = None
        self.running = True
        
    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ╔═╗ ╦ ╦ ╔═╗ ╦═╗ ╔═╗    QuARA - Quantitative Academic Research Agent     ║
║    ║═╬╗║ ║ ╠═╣ ╠╦╝ ╠═╣    Multi-Agent Research Assistant                   ║
║    ╚═╝╚╚═╝ ╩ ╩ ╩╚═ ╩ ╩    Version 1.0                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Welcome to QuARA! Type 'help' for available commands or 'exit' to quit.
"""
        print(banner)
    
    def print_help(self):
        """Print help message"""
        help_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║ Available Commands                                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Research Commands: ⚠️  Note: Full workflow is simulated (no real artifacts yet)
  research <question>     - Simulate full research workflow
  auto-research <question> - 🔥 AUTOMATED: Download data, run code, create plots!
  validate <question>     - Validate a research question
  estimate <question>     - Estimate time for research

LLM Commands: ✅ Fully Functional
  chat                    - Start chat session with LLM
  ask <question>          - Ask a single question to LLM
  code <task>             - Generate code (uses Qwen-Coder)
  write <topic>           - Generate academic content (uses DeepSeek)
  analyze <topic>         - Get detailed analysis with reasoning

System Commands:
  config                  - Show system configuration
  models                  - Show available models
  test                    - Test LLM connection
  clear                   - Clear screen

General:
  help                    - Show this help message
  exit/quit               - Exit QuARA CLI

💡 Debug Mode:
  Use --debug flag to enable detailed execution logging:
    python quara/cli.py --debug
  Then run auto-research to see all debug logs during execution.

Recommended Commands (Working):
  auto-research <question> 🔥 - Fully automated data + code + plots!
  ask Explain the US-China relationship's impact on financial markets
  code Write Python to analyze SPY returns correlation with geopolitical events
  write Literature review on geopolitical risk and stock market returns
  chat (for interactive discussion)

Examples:
  python quara/cli.py --debug    # Enable debug mode (interactive)
  ask What factors in US-China relations affect stock markets?
  code Create a function to download and analyze SPY historical data
  write Methods section for geopolitical risk analysis study
"""
        print(help_text)
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*80)
        print("  QuARA System Configuration")
        print("="*80)
        
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if api_key:
            print(f"\n✅ API Key: {api_key[:15]}...{api_key[-4:]}")
        else:
            print("\n❌ API Key: Not configured")
        
        print(f"\n🌐 Provider Settings:")
        print(f"   Provider: {os.getenv('DEFAULT_LLM_PROVIDER', 'siliconflow')}")
        print(f"   API Base: {os.getenv('QUARA_API_BASE', 'https://api.siliconflow.cn/v1')}")
        
        print(f"\n🤖 Model Configuration:")
        print(f"   Reasoning: {os.getenv('REASONING_MODEL', 'deepseek-ai/DeepSeek-V3.2-Exp')}")
        print(f"   Code:      {os.getenv('CODE_MODEL', 'Qwen/Qwen2.5-Coder-32B-Instruct')}")
        print(f"   Writing:   {os.getenv('WRITING_MODEL', 'deepseek-ai/DeepSeek-V3.2-Exp')}")
        
        print(f"\n⚙️  Parameters:")
        print(f"   Timeout: {os.getenv('QUARA_TIMEOUT', '180')}s")
        print(f"   Max Retries: {os.getenv('QUARA_MAX_RETRIES', '3')}")
        print(f"   Project Root: {os.getenv('QUARA_PROJECT_ROOT', './research_projects')}")
        print()
    
    async def test_connection(self):
        """Test LLM connection"""
        print("\n🔧 Testing LLM connection...")
        
        try:
            if not self.llm_client:
                self.llm_client = create_llm_client()
            
            print("✅ LLM client initialized")
            
            # Test simple generation
            print("\n📡 Sending test request...")
            response = await self.llm_client.generate(
                "Say 'Hello from QuARA!' in one sentence.",
                temperature=0.5
            )
            
            print(f"\n✅ Connection successful!")
            print(f"📝 Response: {response}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Connection failed: {e}")
            return False
    
    async def validate_question(self, question: str):
        """Validate research question"""
        print(f"\n🔍 Validating: {question}")
        
        result = validate_research_request(question)
        
        print(f"\n{'✅' if result['valid'] else '❌'} Validation Result:")
        print(f"   Valid: {result['valid']}")
        print(f"   Word Count: {result.get('word_count', 0)}")
        print(f"   Has Research Indicators: {result.get('has_research_indicators', False)}")
        
        if result['valid']:
            time_est = estimate_research_time(question)
            print(f"\n⏱️  Estimated Time: {time_est}")
    
    async def conduct_research(self, question: str):
        """Conduct research (currently simulated)"""
        print(f"\n🔬 Research Request: {question}")
        print("="*80)
        print("\n⚠️  Note: Full research workflow is currently simulated.")
        print("    Real artifact generation is under development.")
        print("\n💡 For working analysis, try:")
        print(f"   • ask {question[:50]}...")
        print(f"   • analyze {question[:50]}...")
        print(f"   • code <programming task related to your question>")
        print("\nWould you like to:")
        print("  1. Continue with simulation")
        print("  2. Get comprehensive analysis instead (recommended)")
        print("  3. Cancel")
        
        # For non-interactive use, proceed with simulation
        print("\n📦 Initializing QuARA system...")
        
        try:
            # Initialize system if needed
            if not self.system:
                self.system = QuARASystem()
                await self.system.start_system()
            
            # Conduct research
            print("\n🚀 Executing research workflow (simulated)...")
            result = await self.system.conduct_research(question)
            
            # Display results
            print("\n" + "="*80)
            print("  Research Results (Simulated)")
            print("="*80)
            
            if result['success']:
                print(f"\n✅ Workflow simulation completed!")
                print(f"📁 Project ID: {result['project_id']}")
                
                # Show project details
                if result.get('result'):
                    res = result['result']
                    if 'directory' in res:
                        print(f"📂 Directory: {res['directory']}")
                    if 'artifacts' in res:
                        print(f"📄 Artifacts (simulated): {len(res['artifacts'])}")
                
                print(f"\n⚠️  Note: No actual files were created (simulation mode)")
                print(f"\n💡 For real analysis, use:")
                print(f"   • analyze {question[:50]}...")
                print(f"   • chat (then discuss your research question)")
                
            else:
                print(f"\n❌ Simulation failed")
                if result.get('error'):
                    print(f"Error: {result['error']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def chat_mode(self):
        """Interactive chat mode"""
        print("\n💬 Entering chat mode (type 'exit' to return)")
        print("="*80)
        
        if not self.llm_client:
            self.llm_client = create_llm_client()
        
        messages = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'back']:
                    print("Exiting chat mode...\n")
                    break
                
                if not user_input:
                    continue
                
                messages.append({"role": "user", "content": user_input})
                
                print("🤖 QuARA: ", end="", flush=True)
                
                response = await self.llm_client.chat_completion(
                    messages=messages,
                    purpose=ModelPurpose.REASONING
                )
                
                print(response['content'])
                
                messages.append({"role": "assistant", "content": response['content']})
                
            except KeyboardInterrupt:
                print("\n\nExiting chat mode...\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    async def ask_question(self, question: str):
        """Ask a single question"""
        if not self.llm_client:
            self.llm_client = create_llm_client()
        
        print(f"\n❓ Question: {question}")
        print("\n🤖 Answer: ", end="", flush=True)
        
        response = await self.llm_client.generate(question)
        print(response)
        print()
    
    async def generate_code(self, task: str):
        """Generate code for a task"""
        if not self.llm_client:
            self.llm_client = create_llm_client()
        
        print(f"\n💻 Generating code for: {task}")
        print("="*80)
        
        code = await self.llm_client.generate_code(
            prompt=task,
            language="python"
        )
        
        print(f"\n{code}")
        print("\n" + "="*80)
    
    async def generate_writing(self, topic: str):
        """Generate academic writing"""
        if not self.llm_client:
            self.llm_client = create_llm_client()
        
        print(f"\n✍️  Generating content about: {topic}")
        print("="*80)
        
        content = await self.llm_client.write_content(
            prompt=topic,
            content_type="academic"
        )
        
        print(f"\n{content}")
        print("\n" + "="*80)
    
    async def analyze_topic(self, topic: str):
        """Perform comprehensive analysis of a topic"""
        if not self.llm_client:
            self.llm_client = create_llm_client()
        
        print(f"\n🔍 Comprehensive Analysis: {topic}")
        print("="*80)
        
        # Multi-stage analysis
        stages = [
            ("📚 Background & Context", "Provide background and context for: "),
            ("🎯 Key Factors", "Identify and explain the key factors related to: "),
            ("📊 Data & Evidence", "What data and evidence are relevant for analyzing: "),
            ("💡 Insights & Implications", "Provide insights and implications about: ")
        ]
        
        for stage_name, prompt_prefix in stages:
            print(f"\n{stage_name}")
            print("-" * 80)
            
            response = await self.llm_client.generate(
                f"{prompt_prefix}{topic}",
                temperature=0.7,
                purpose=ModelPurpose.REASONING
            )
            
            print(response)
        
        print("\n" + "="*80)
        print("✅ Analysis complete!")
        print()
    
    async def automated_research(self, question: str):
        """Execute fully automated research with data download and code execution"""
        print(f"\n🚀 AUTOMATED RESEARCH: {question}")
        print("="*80)
        print("\n✨ This will:")
        print("  1. Plan the research and identify data needs")
        print("  2. Download required financial data automatically")
        print("  3. Generate analysis code using LLM")
        print("  4. Execute the code and run statistical analysis")
        print("  5. Create visualizations (plots, charts)")
        print("  6. Generate a comprehensive research report")
        print("\n⏱️  This may take 2-5 minutes...")
        print()
        
        try:
            # Initialize automated executor
            if not self.auto_executor:
                self.auto_executor = AutomatedResearchExecutor()
            
            # Execute automated research
            print("🔬 Starting automated workflow...")
            results = await self.auto_executor.execute_research(question)
            
            # Display results
            print("\n" + "="*80)
            print("  AUTOMATED RESEARCH RESULTS")
            print("="*80)
            
            if results.get("success"):
                print(f"\n✅ Research completed successfully!")
                print(f"\n📁 Project: {results['project_id']}")
                print(f"📂 Directory: {results['directory']}")
                
                # Show what was created
                stages = results.get("stages", {})
                
                if "data_collection" in stages:
                    datasets = stages["data_collection"].get("datasets", {})
                    print(f"\n📊 Data Downloaded: {len(datasets)} datasets")
                    for ticker, info in datasets.items():
                        print(f"   • {ticker}: {info['rows']} rows ({info['date_range']})")
                
                if "analysis" in stages:
                    analysis = stages["analysis"]
                    if analysis["execution"]["success"]:
                        print(f"\n✅ Analysis Code: Generated and executed successfully")
                        print(f"   Code: {analysis['code_file']}")
                    else:
                        print(f"\n⚠️  Analysis execution had issues")
                        if analysis["execution"]["stderr"]:
                            print(f"   Error: {analysis['execution']['stderr'][:200]}")
                
                if "visualizations" in stages:
                    viz = stages["visualizations"]
                    plots = viz.get("plots", [])
                    print(f"\n📈 Basic Visualizations: {len(plots)} plots created")
                    for plot in plots[:5]:  # Show first 5
                        print(f"   • {plot}")
                    if len(plots) > 5:
                        print(f"   ... and {len(plots) - 5} more")
                    
                    if not plots and viz.get("execution", {}).get("error"):
                        print(f"\n⚠️  Visualization generation had errors:")
                        print(f"   {viz['execution']['error'][:200]}")
                
                if "section_visualizations" in stages:
                    section_viz = stages["section_visualizations"]
                    section_plots = section_viz.get("plots", [])
                    print(f"\n🎨 Section-Specific Visualizations: {len(section_plots)} plots created")
                    for plot in section_plots:
                        print(f"   • {plot}")
                    
                    if not section_plots and section_viz.get("execution", {}).get("error"):
                        print(f"\n⚠️  Section visualization generation had errors:")
                        print(f"   {section_viz['execution']['error'][:200]}")
                
                if "report" in stages:
                    report = stages["report"]
                    print(f"\n📝 Report: {report['file']}")
                    print(f"\nPreview:")
                    print("-" * 80)
                    print(report.get("preview", ""))
                
                print(f"\n💡 Next steps:")
                print(f"   • Open directory: cd {results['directory']}")
                print(f"   • View plots: open {results['directory']}/*.png")
                print(f"   • Read report: cat {results['directory']}/research_report.md")
                
            else:
                print(f"\n❌ Research failed")
                if results.get("error"):
                    print(f"Error: {results['error']}")
            
        except Exception as e:
            print(f"\n❌ Automated research failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def list_projects(self):
        """List all research projects"""
        project_root = os.getenv('QUARA_PROJECT_ROOT', './research_projects')
        
        print(f"\n📁 Research Projects in {project_root}:")
        print("="*80)
        
        if not os.path.exists(project_root):
            print("\nNo projects found. Start a new research project with 'research <question>'")
            return
        
        projects = [d for d in os.listdir(project_root) 
                   if os.path.isdir(os.path.join(project_root, d))]
        
        if not projects:
            print("\nNo projects found.")
        else:
            for i, project in enumerate(projects, 1):
                project_path = os.path.join(project_root, project)
                files = len([f for f in os.listdir(project_path) 
                           if os.path.isfile(os.path.join(project_path, f))])
                print(f"{i}. {project} ({files} files)")
        print()
    
    async def show_project_status(self, project_id: Optional[str] = None):
        """Show project status"""
        if not project_id:
            print("\n❌ Please provide a project ID")
            print("Usage: status <project_id>")
            return
        
        if not self.system:
            self.system = QuARASystem()
            await self.system.start_system()
        
        print(f"\n📊 Project Status: {project_id}")
        print("="*80)
        
        try:
            status = await self.system.get_project_status(project_id)
            
            print(f"\n{'✅' if status['project_exists'] else '❌'} Exists: {status['project_exists']}")
            print(f"📂 Directory: {status['project_directory']}")
            
            if status['artifacts']:
                print(f"\n📄 Artifacts ({len(status['artifacts'])}):")
                for artifact in status['artifacts']:
                    print(f"   - {artifact}")
            
            if status.get('memory_summary'):
                mem = status['memory_summary']
                print(f"\n🧠 Memory:")
                print(f"   Nodes: {mem.get('node_count', 0)}")
                print(f"   Types: {', '.join(mem.get('node_types', []))}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
        print()
    
    def show_models(self):
        """Show available models"""
        print("\n🤖 Available Models")
        print("="*80)
        
        print("\n📋 Configured Models:")
        print(f"   🧠 Reasoning: deepseek-ai/DeepSeek-V3.2-Exp")
        print(f"      - Agent thinking, tool selection, decision making")
        print(f"      - Temperature: 0.7")
        
        print(f"\n   💻 Code: Qwen/Qwen2.5-Coder-32B-Instruct")
        print(f"      - Python/R/SQL generation, statistical analysis")
        print(f"      - Temperature: 0.3")
        
        print(f"\n   ✍️  Writing: deepseek-ai/DeepSeek-V3.2-Exp")
        print(f"      - Academic writing, documentation")
        print(f"      - Temperature: 0.8")
        
        print("\n📚 Other Available Models:")
        print("   - deepseek-ai/DeepSeek-V2.5")
        print("   - deepseek-ai/DeepSeek-Coder")
        print("   - Qwen/Qwen2.5-72B-Instruct")
        print()
    
    async def process_command(self, command: str):
        """Process a user command"""
        if not command.strip():
            return
        
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd in ['exit', 'quit']:
                self.running = False
                print("\n👋 Thank you for using QuARA! Goodbye.\n")
                
            elif cmd == 'help':
                self.print_help()
                
            elif cmd == 'clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                self.print_banner()
                
            elif cmd == 'config':
                self.print_config()
                
            elif cmd == 'models':
                self.show_models()
                
            elif cmd == 'test':
                await self.test_connection()
                
            elif cmd == 'research':
                if not args:
                    print("\n❌ Please provide a research question")
                    print("Usage: research <your research question>")
                else:
                    await self.conduct_research(args)
            
            elif cmd == 'auto-research':
                if not args:
                    print("\n❌ Please provide a research question")
                    print("Usage: auto-research <your research question>")
                else:
                    await self.automated_research(args)
                    
            elif cmd == 'validate':
                if not args:
                    print("\n❌ Please provide a research question")
                else:
                    await self.validate_question(args)
                    
            elif cmd == 'estimate':
                if not args:
                    print("\n❌ Please provide a research question")
                else:
                    time_est = estimate_research_time(args)
                    print(f"\n⏱️  Estimated time: {time_est}")
                    
            elif cmd == 'chat':
                await self.chat_mode()
                
            elif cmd == 'ask':
                if not args:
                    print("\n❌ Please provide a question")
                else:
                    await self.ask_question(args)
                    
            elif cmd == 'code':
                if not args:
                    print("\n❌ Please describe what code to generate")
                else:
                    await self.generate_code(args)
                    
            elif cmd == 'write':
                if not args:
                    print("\n❌ Please provide a topic to write about")
                else:
                    await self.generate_writing(args)
            
            elif cmd == 'analyze':
                if not args:
                    print("\n❌ Please provide a topic to analyze")
                else:
                    await self.analyze_topic(args)
                    
            elif cmd == 'list':
                await self.list_projects()
                
            elif cmd == 'status':
                await self.show_project_status(args if args else None)
                
            else:
                print(f"\n❌ Unknown command: {cmd}")
                print("Type 'help' for available commands")
                
        except Exception as e:
            print(f"\n❌ Error executing command: {e}")
            import traceback
            traceback.print_exc()
    
    async def run_interactive(self):
        """Run interactive mode"""
        self.print_banner()
        
        # Check configuration
        if not os.getenv("SILICONFLOW_API_KEY"):
            print("⚠️  Warning: SILICONFLOW_API_KEY not set")
            print("Please configure your API key in .env file\n")
        
        while self.running:
            try:
                command = input("QuARA> ").strip()
                if command:
                    await self.process_command(command)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except EOFError:
                print("\n\n👋 Goodbye!\n")
                break
        
        # Cleanup
        if self.system:
            await self.system.stop_system()
    
    async def run_command(self, args):
        """Run single command from CLI args"""
        if args.command == 'research':
            await self.conduct_research(args.question)
        elif args.command == 'chat':
            await self.chat_mode()
        elif args.command == 'config':
            self.print_config()
        elif args.command == 'test':
            await self.test_connection()
        elif args.command == 'models':
            self.show_models()
        
        # Cleanup
        if self.system:
            await self.system.stop_system()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='QuARA - Quantitative Academic Research Agent CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add global --debug flag
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging for detailed execution traces'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Research command
    research_parser = subparsers.add_parser('research', help='Conduct research')
    research_parser.add_argument('question', help='Research question')
    
    # Chat command
    subparsers.add_parser('chat', help='Start chat session')
    
    # Config command
    subparsers.add_parser('config', help='Show configuration')
    
    # Test command
    subparsers.add_parser('test', help='Test LLM connection')
    
    # Models command
    subparsers.add_parser('models', help='Show available models')
    
    args = parser.parse_args()
    
    # Configure logging based on --debug flag
    import logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(levelname)s:%(name)s:%(message)s'
        )
        print("🐛 DEBUG MODE ENABLED - Detailed logging active\n")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s:%(message)s'
        )
    
    cli = QuARACLI()
    
    if args.command:
        # Run specific command
        await cli.run_command(args)
    else:
        # Run interactive mode
        await cli.run_interactive()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
