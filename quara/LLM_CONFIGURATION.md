# QuARA LLM Configuration Guide

## Using SiliconFlow with DeepSeek-V3

QuARA now supports configurable LLM providers, with built-in support for **SiliconFlow** and **DeepSeek models**.

## Quick Start

### 1. Get Your API Key

1. Visit [SiliconFlow](https://siliconflow.cn)
2. Create an account and get your API key
3. Note: SiliconFlow provides access to DeepSeek-V3 and other models

### 2. Set Up Environment

**Option A: Environment Variables (Recommended)**

```bash
export SILICONFLOW_API_KEY="your_api_key_here"
export QUARA_LLM_MODEL="deepseek-ai/DeepSeek-V3"
export QUARA_LLM_PROVIDER="siliconflow"
```

**Option B: Create `.env` File**

Copy `.env.example` to `.env` and fill in your settings:

```bash
cp .env.example .env
# Edit .env with your API key
```

### 3. Use QuARA

```python
import asyncio
from quara import QuARASystem

async def main():
    # System automatically reads from environment
    async with QuARASystem() as system:
        result = await system.conduct_research(
            "Analyze the effect of meditation on stress levels"
        )
        print(f"Success: {result['success']}")
        print(f"Project ID: {result['project_id']}")

asyncio.run(main())
```

## Configuration Methods

### Method 1: Environment Variables (Simplest)

```bash
# Required
export SILICONFLOW_API_KEY="sk-your-key-here"

# Optional (with defaults)
export QUARA_LLM_PROVIDER="siliconflow"  # Default: siliconflow
export QUARA_LLM_MODEL="deepseek-ai/DeepSeek-V3"  # Default: DeepSeek-V3
export QUARA_TEMPERATURE="0.7"  # Default: 0.7
export QUARA_MAX_TOKENS="4096"  # Default: 4096
```

```python
from quara import QuARASystem

async with QuARASystem() as system:
    result = await system.conduct_research("Your research question")
```

### Method 2: Configuration Dictionary

```python
from quara import QuARASystem

llm_config = {
    "provider": "siliconflow",
    "model": "deepseek-ai/DeepSeek-V3",
    "api_key": "your_api_key",
    "temperature": 0.7,
    "max_tokens": 4096
}

async with QuARASystem(llm_config=llm_config) as system:
    result = await system.conduct_research("Your research question")
```

### Method 3: LLM Client Object

```python
from quara import QuARASystem
from quara.utils import create_llm_client

# Create and configure LLM client
llm_client = create_llm_client(
    provider="siliconflow",
    model="deepseek-ai/DeepSeek-V3",
    api_key="your_api_key",
    temperature=0.8,
    max_tokens=8192
)

# Test LLM directly
response = await llm_client.generate(
    prompt="What is causal inference?",
    system_prompt="You are a research methodology expert."
)
print(response)

# Use with QuARA
async with QuARASystem(llm_client=llm_client) as system:
    result = await system.conduct_research("Your research question")
```

### Method 4: Advanced Configuration

```python
from quara.utils import LLMClient, LLMConfig, LLMProvider

# Create detailed configuration
config = LLMConfig(
    provider=LLMProvider.SILICONFLOW,
    model="deepseek-ai/DeepSeek-V3",
    api_key="your_api_key",
    api_base="https://api.siliconflow.cn/v1",
    temperature=0.7,
    max_tokens=4096,
    timeout=120,  # 2 minutes
    max_retries=5
)

llm_client = LLMClient(config)

async with QuARASystem(llm_client=llm_client) as system:
    result = await system.conduct_research("Your research question")
```

## Available Models

SiliconFlow supports multiple DeepSeek models:

- **`deepseek-ai/DeepSeek-V3`** - Latest and most capable (Recommended)
- **`deepseek-ai/DeepSeek-V2.5`** - Previous version
- **`deepseek-ai/DeepSeek-Coder`** - Optimized for code generation
- **`deepseek-ai/DeepSeek-Chat`** - Optimized for conversation

Check [SiliconFlow documentation](https://siliconflow.cn/docs) for the latest available models.

## LLM Client API

### Basic Usage

```python
from quara.utils import create_llm_client

# Create client
client = create_llm_client(
    provider="siliconflow",
    model="deepseek-ai/DeepSeek-V3",
    api_key="your_key"
)

# Simple generation
response = await client.generate(
    prompt="Explain random forests",
    temperature=0.5
)

# Chat completion
result = await client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
)

# Structured JSON output
data = await client.structured_output(
    prompt="Design a research study about sleep",
    schema={
        "title": "string",
        "hypothesis": "string",
        "methods": ["string"]
    }
)
```

### Configuration Management

```python
# Get current config
config = client.get_config()
print(config)

# Update configuration
client.update_config(
    temperature=0.8,
    max_tokens=8192
)
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SILICONFLOW_API_KEY` | - | Your SiliconFlow API key (required) |
| `QUARA_API_KEY` | - | Alternative generic key name |
| `QUARA_LLM_PROVIDER` | `siliconflow` | LLM provider: siliconflow, openai, custom |
| `QUARA_LLM_MODEL` | `deepseek-ai/DeepSeek-V3` | Model name |
| `QUARA_API_BASE` | `https://api.siliconflow.cn/v1` | API endpoint URL |
| `QUARA_TEMPERATURE` | `0.7` | Sampling temperature (0.0-2.0) |
| `QUARA_MAX_TOKENS` | `4096` | Maximum tokens in response |
| `QUARA_TIMEOUT` | `60` | Request timeout in seconds |
| `QUARA_MAX_RETRIES` | `3` | Maximum retry attempts |

## Using Other Providers

### OpenAI

```python
llm_config = {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your_openai_key"
}

async with QuARASystem(llm_config=llm_config) as system:
    result = await system.conduct_research("Your question")
```

### Custom OpenAI-Compatible Endpoint

```python
llm_config = {
    "provider": "custom",
    "model": "custom-model-name",
    "api_key": "your_key",
    "api_base": "https://your-endpoint.com/v1"
}

async with QuARASystem(llm_config=llm_config) as system:
    result = await system.conduct_research("Your question")
```

## Troubleshooting

### API Key Issues

```bash
# Check if key is set
echo $SILICONFLOW_API_KEY

# Set temporarily
export SILICONFLOW_API_KEY="your_key"

# Set permanently (add to ~/.zshrc or ~/.bashrc)
echo 'export SILICONFLOW_API_KEY="your_key"' >> ~/.zshrc
source ~/.zshrc
```

### Import Errors

```bash
# Install OpenAI package (required for SiliconFlow)
pip install openai

# Or install all requirements
pip install -r requirements.txt
```

### Connection Issues

1. Check your internet connection
2. Verify API key is correct
3. Check SiliconFlow service status
4. Try increasing timeout:

```python
llm_config = {
    "provider": "siliconflow",
    "timeout": 180,  # 3 minutes
    "max_retries": 5
}
```

### Running Without LLM

QuARA can run in demo mode without a real LLM:

```python
# System will detect missing API key and use mock responses
async with QuARASystem() as system:
    result = await system.conduct_research("Your question")
```

## Examples

See detailed examples in:
- `examples/siliconflow_config_example.py` - Comprehensive configuration examples
- `examples/simple_usage.py` - Basic usage
- `examples/complete_research_example.py` - Full workflow

Run an example:

```bash
# Set your API key
export SILICONFLOW_API_KEY="your_key"

# Run example
python examples/siliconflow_config_example.py
```

## Best Practices

1. **Use Environment Variables** for production deployments
2. **Use `.env` files** for local development (don't commit to git!)
3. **Set appropriate timeouts** for long-running research tasks
4. **Monitor token usage** using the LLM client's response metadata
5. **Handle errors gracefully** with try-except blocks
6. **Use structured output** for parseable responses

## Security Notes

- ⚠️ Never commit API keys to version control
- ✅ Use `.env` files (add to `.gitignore`)
- ✅ Use environment variables in production
- ✅ Rotate API keys regularly
- ✅ Use separate keys for dev/prod environments

## Performance Tips

1. **Adjust temperature**: Lower (0.3-0.5) for factual tasks, higher (0.7-0.9) for creative tasks
2. **Set max_tokens wisely**: More tokens = longer responses but higher cost
3. **Use caching**: QuARA's memory system reduces redundant LLM calls
4. **Batch operations**: Let the system handle multiple agent calls efficiently

## Support

- SiliconFlow Documentation: https://siliconflow.cn/docs
- QuARA Issues: https://github.com/yourusername/quara/issues
- DeepSeek Documentation: https://deepseek.com/docs

## License

This configuration is part of the QuARA project and follows the same license.
