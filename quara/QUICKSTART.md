# 🚀 QuARA Quick Start with SiliconFlow + DeepSeek-V3

## 30-Second Setup

```bash
# 1. Get API key from https://siliconflow.cn
# 2. Set environment variable
export SILICONFLOW_API_KEY="your_api_key_here"

# 3. Run QuARA
python -c "
import asyncio
from quara import QuARASystem

async def main():
    async with QuARASystem() as system:
        result = await system.conduct_research(
            'Analyze the effect of exercise on mental health'
        )
        print(f'Success: {result[\"success\"]}')
        print(f'Project: {result[\"project_id\"]}')

asyncio.run(main())
"
```

## Configuration Methods

### 🌟 Method 1: Environment Variables (Easiest)
```bash
export SILICONFLOW_API_KEY="sk-xxx"
export QUARA_LLM_MODEL="deepseek-ai/DeepSeek-V3"
```

### 📝 Method 2: .env File
```bash
# Create .env file
echo "SILICONFLOW_API_KEY=your_key" > .env
echo "QUARA_LLM_MODEL=deepseek-ai/DeepSeek-V3" >> .env
```

### 🔧 Method 3: Config Dict
```python
llm_config = {
    "provider": "siliconflow",
    "model": "deepseek-ai/DeepSeek-V3",
    "api_key": "your_key"
}
system = QuARASystem(llm_config=llm_config)
```

### ⚡ Method 4: Pre-configured Client
```python
from quara.utils import create_llm_client

client = create_llm_client(
    provider="siliconflow",
    model="deepseek-ai/DeepSeek-V3",
    api_key="your_key"
)
system = QuARASystem(llm_client=client)
```

## Common Usage Patterns

### Simple Research
```python
from quara import conduct_research

result = await conduct_research(
    "What is the relationship between coffee and productivity?"
)
```

### Full Control
```python
from quara import QuARASystem

async with QuARASystem() as system:
    result = await system.conduct_research(
        "Your research question",
        context={"domain": "psychology", "method": "meta-analysis"}
    )
```

### Direct LLM Access
```python
from quara.utils import create_llm_client

client = create_llm_client()
response = await client.generate("Explain causal inference")
```

## Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `deepseek-ai/DeepSeek-V3` | Latest, most capable | **Recommended for research** |
| `deepseek-ai/DeepSeek-V2.5` | Stable version | General use |
| `deepseek-ai/DeepSeek-Coder` | Code-optimized | Data analysis |
| `deepseek-ai/DeepSeek-Chat` | Chat-optimized | Interactive tasks |

## Test Your Setup

```bash
# Quick test
python tests/test_llm_config.py

# Interactive examples
python examples/siliconflow_config_example.py

# Full research workflow
python examples/complete_research_example.py
```

## Troubleshooting

### No API Key Error
```bash
# Check if key is set
echo $SILICONFLOW_API_KEY

# Set it
export SILICONFLOW_API_KEY="your_key"

# Or add to your shell profile
echo 'export SILICONFLOW_API_KEY="your_key"' >> ~/.zshrc
source ~/.zshrc
```

### Import Error
```bash
pip install openai python-dotenv
```

### Connection Issues
```python
# Increase timeout
llm_config = {
    "provider": "siliconflow",
    "timeout": 180,  # 3 minutes
    "max_retries": 5
}
```

## Configuration Parameters

```python
{
    "provider": "siliconflow",          # LLM provider
    "model": "deepseek-ai/DeepSeek-V3", # Model name
    "api_key": "your_key",              # API key
    "api_base": "https://...",          # Custom endpoint (optional)
    "temperature": 0.7,                 # 0.0-2.0, creativity level
    "max_tokens": 4096,                 # Max response length
    "timeout": 60,                      # Seconds
    "max_retries": 3                    # Retry attempts
}
```

## Get Help

- 📖 Full Guide: [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md)
- 🔍 Examples: `examples/siliconflow_config_example.py`
- ✅ Test: `tests/test_llm_config.py`
- 🌐 API Keys: https://siliconflow.cn
- 📚 DeepSeek: https://deepseek.com/docs

## Quick Commands

```bash
# Set API key
export SILICONFLOW_API_KEY="your_key"

# Test configuration
python tests/test_llm_config.py

# Run example
python examples/siliconflow_config_example.py

# Start research
python -c "
import asyncio
from quara import conduct_research
print(asyncio.run(conduct_research('Your question')))
"
```

---

**🎯 That's it! You're ready to use QuARA with SiliconFlow + DeepSeek-V3!**

For detailed documentation, see [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md)
