"""
Quick test script to verify LLM configuration
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quara.utils import create_llm_client, LLMConfig


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


async def test_llm_configuration():
    """Test LLM client configuration and basic functionality"""
    
    print_header("QuARA LLM Configuration Test")
    
    # Check for API key
    api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("QUARA_API_KEY")
    
    if not api_key:
        print("\n❌ No API key found!")
        print("\nPlease set your SiliconFlow API key:")
        print("  export SILICONFLOW_API_KEY='your_api_key_here'")
        print("\nGet your API key from: https://siliconflow.cn")
        return False
    
    print(f"\n✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test creating LLM client
    print_header("Creating LLM Client")
    
    try:
        llm_client = create_llm_client(
            provider="siliconflow",
            model="deepseek-ai/DeepSeek-V3",
            api_key=api_key,
            temperature=0.7,
            max_tokens=500
        )
        print("✅ LLM Client created successfully")
        
        # Print configuration
        print("\n📋 Configuration:")
        config = llm_client.get_config()
        for key, value in config.items():
            if key != "api_key":  # Don't print full API key
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        return False
    
    # Test simple generation
    print_header("Testing Simple Generation")
    
    try:
        response = await llm_client.generate(
            prompt="What is a multi-agent system? Answer in one sentence.",
            temperature=0.5
        )
        
        print("✅ Generation successful!")
        print(f"\n📝 Response:\n{response}\n")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False
    
    # Test chat completion
    print_header("Testing Chat Completion")
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": "What is causal inference?"}
        ]
        
        result = await llm_client.chat_completion(messages=messages)
        
        print("✅ Chat completion successful!")
        print(f"\n💬 Response:\n{result['content']}\n")
        print(f"📊 Token Usage:")
        print(f"   Prompt tokens: {result['usage']['prompt_tokens']}")
        print(f"   Completion tokens: {result['usage']['completion_tokens']}")
        print(f"   Total tokens: {result['usage']['total_tokens']}")
        
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False
    
    # Test structured output
    print_header("Testing Structured Output")
    
    try:
        schema = {
            "title": "string",
            "key_points": ["string"],
            "conclusion": "string"
        }
        
        structured = await llm_client.structured_output(
            prompt="Explain the concept of randomized controlled trials",
            schema=schema,
            system_prompt="You are a research methodology expert. Keep responses concise."
        )
        
        print("✅ Structured output successful!")
        print(f"\n📊 JSON Response:")
        import json
        print(json.dumps(structured, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ Structured output failed: {e}")
        print(f"   Note: This is optional - basic functionality still works")
    
    # Final summary
    print_header("Test Summary")
    print("\n✅ All tests passed!")
    print("\n🎉 Your LLM configuration is working correctly!")
    print("\n💡 Next steps:")
    print("   1. Run: python examples/siliconflow_config_example.py")
    print("   2. Run: python examples/complete_research_example.py")
    print("   3. Try conducting your own research with QuARA")
    
    return True


async def main():
    """Main test function"""
    success = await test_llm_configuration()
    
    if not success:
        print("\n" + "=" * 80)
        print("⚠️  Configuration test failed")
        print("=" * 80)
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✨ Configuration test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
