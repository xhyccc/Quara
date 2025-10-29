"""
Quick Configuration Verification Test
Verifies that different models are configured for different purposes
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables FIRST before importing quara modules
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    loaded = load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
    if not loaded:
        print("⚠️  Warning: .env file not found or could not be loaded")
except ImportError:
    print("⚠️  python-dotenv not installed")

# Now import quara modules AFTER env vars are loaded
from quara.utils import create_llm_client, ModelPurpose

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_section("🔧 QuARA Multi-Model Configuration Verification")
    
    # Check API key
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("\n❌ SILICONFLOW_API_KEY not set!")
        return False
    
    print(f"\n✅ API Key: {api_key[:15]}...{api_key[-4:]}")
    
    # Debug environment variables
    print("\n🔍 Debug - Environment Variables:")
    print(f"   DEFAULT_MODEL: {os.getenv('DEFAULT_MODEL')}")
    print(f"   REASONING_MODEL: {os.getenv('REASONING_MODEL')}")
    print(f"   CODE_MODEL: {os.getenv('CODE_MODEL')}")
    print(f"   WRITING_MODEL: {os.getenv('WRITING_MODEL')}")
    
    # Create client
    print("\n📦 Creating LLM Client...")
    client = create_llm_client()
    
    # Show configuration
    print_section("⚙️  Client Configuration")
    config = client.get_config()
    
    # Debug: print raw config object
    print("\n🔍 Debug - Raw Config Object:")
    print(f"   config.model: {client.config.model}")
    print(f"   config.reasoning_model: {client.config.reasoning_model}")
    print(f"   config.code_model: {client.config.code_model}")
    print(f"   config.writing_model: {client.config.writing_model}")
    
    print("\n🌐 Provider Settings:")
    print(f"   Provider: {config['provider']}")
    print(f"   API Base: {config['api_base']}")
    print(f"   Timeout: {config['timeout']}s")
    print(f"   Max Retries: {config['max_retries']}")
    
    print("\n🤖 Model Configuration:")
    print(f"   Default Model: {config['model']}")
    
    # Show specialized models
    print("\n🎯 Specialized Models:")
    if config.get('reasoning_model'):
        print(f"   🧠 Reasoning: {config['reasoning_model']}")
        print(f"      Purpose: Agent thinking, tool calls, decision making")
        print(f"      Temperature: {os.getenv('REASONING_TEMPERATURE', '0.7')}")
    else:
        print(f"   🧠 Reasoning: Using default model")
    
    if config.get('code_model'):
        print(f"\n   💻 Code Generation: {config['code_model']}")
        print(f"      Purpose: Python/R/SQL generation, statistical analysis")
        print(f"      Temperature: {os.getenv('CODE_TEMPERATURE', '0.3')}")
    else:
        print(f"\n   💻 Code Generation: Using default model")
    
    if config.get('writing_model'):
        print(f"\n   ✍️  Writing: {config['writing_model']}")
        print(f"      Purpose: Manuscript generation, documentation")
        print(f"      Temperature: {os.getenv('WRITING_TEMPERATURE', '0.8')}")
    else:
        print(f"\n   ✍️  Writing: Using default model")
    
    # Test model selection
    print_section("🔍 Model Selection Test")
    
    reasoning_model = client._get_model_for_purpose(ModelPurpose.REASONING)
    code_model = client._get_model_for_purpose(ModelPurpose.CODE)
    writing_model = client._get_model_for_purpose(ModelPurpose.WRITING)
    general_model = client._get_model_for_purpose(ModelPurpose.GENERAL)
    
    print("\n📋 Model Selection by Purpose:")
    print(f"   Reasoning → {reasoning_model}")
    print(f"   Code      → {code_model}")
    print(f"   Writing   → {writing_model}")
    print(f"   General   → {general_model}")
    
    # Verify different models for different purposes
    if code_model != reasoning_model:
        print("\n✅ SUCCESS: Different models configured for reasoning and code!")
        print(f"   Reasoning: {reasoning_model}")
        print(f"   Code: {code_model}")
    else:
        print("\n⚠️  WARNING: Same model used for reasoning and code")
        print("   Consider setting CODE_MODEL to a code-specialized model")
    
    print_section("✨ Configuration Summary")
    print("\n✅ Configuration is valid and ready to use!")
    print("\n📚 Your setup:")
    print(f"   • Reasoning & Decisions: {reasoning_model}")
    print(f"   • Code Generation: {code_model}")
    print(f"   • Writing: {writing_model}")
    
    print("\n🚀 Next steps:")
    print("   1. Run: python examples/multi_model_example.py")
    print("   2. Test specific model: Choose option 1, 2, or 3")
    print("   3. Run full workflow: Choose option 5")
    
    print("\n" + "=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
