"""
LLM Client for QuARA - Configurable LLM interface
Supports multiple providers including SiliconFlow with DeepSeek models
"""

import os
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
from dataclasses import dataclass
import json

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    SILICONFLOW = "siliconflow"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class ModelPurpose(str, Enum):
    """Different purposes requiring different models"""
    REASONING = "reasoning"  # Agent thinking, tool calls, decisions
    CODE = "code"  # Code generation, statistical analysis
    WRITING = "writing"  # Manuscript generation, documentation
    GENERAL = "general"  # Default fallback


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    provider: LLMProvider = LLMProvider.SILICONFLOW
    model: str = "deepseek-ai/DeepSeek-V3.2-Exp"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3
    
    # Model-specific configurations
    reasoning_model: Optional[str] = None
    code_model: Optional[str] = None
    writing_model: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables"""
        provider = os.getenv("DEFAULT_LLM_PROVIDER") or os.getenv("QUARA_LLM_PROVIDER", "siliconflow")
        model = os.getenv("DEFAULT_MODEL") or os.getenv("QUARA_LLM_MODEL", "deepseek-ai/DeepSeek-V3.2-Exp")
        api_key = (
            os.getenv("SILICONFLOW_API_KEY") or 
            os.getenv("QUARA_API_KEY") or 
            os.getenv("OPENAI_API_KEY") or
            os.getenv("ANTHROPIC_API_KEY")
        )
        api_base = (
            os.getenv("SILICONFLOW_API_ENDPOINT") or
            os.getenv("QUARA_API_BASE") or 
            os.getenv("SILICONFLOW_API_BASE")
        )
        
        # Get model-specific configurations
        reasoning_model = os.getenv("REASONING_MODEL", model)
        code_model = os.getenv("CODE_MODEL") or os.getenv("CODE_LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
        writing_model = os.getenv("WRITING_MODEL", model)
        
        return cls(
            provider=LLMProvider(provider.lower()),
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=float(os.getenv("QUARA_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("QUARA_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("QUARA_TIMEOUT", "60")),
            max_retries=int(os.getenv("QUARA_MAX_RETRIES", "3")),
            reasoning_model=reasoning_model,
            code_model=code_model,
            writing_model=writing_model
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create config from dictionary"""
        if "provider" in config_dict:
            config_dict["provider"] = LLMProvider(config_dict["provider"].lower())
        return cls(**config_dict)


class LLMClient:
    """
    Universal LLM client with support for multiple providers and model purposes
    
    Supports different models for different purposes:
    - Reasoning: Agent thinking, tool calls, decision making (DeepSeek-V3.2)
    - Code: Code generation, statistical analysis (Qwen-Coder)
    - Writing: Manuscript generation, documentation (DeepSeek-V3.2)
    
    Currently supports:
    - SiliconFlow (https://siliconflow.cn) with DeepSeek and Qwen models
    - OpenAI API
    - Custom OpenAI-compatible endpoints
    
    Example:
        # Using SiliconFlow with different models
        config = LLMConfig(
            provider=LLMProvider.SILICONFLOW,
            model="deepseek-ai/DeepSeek-V3.2-Exp",
            reasoning_model="deepseek-ai/DeepSeek-V3.2-Exp",
            code_model="Qwen/Qwen2.5-Coder-32B-Instruct",
            api_key="your_siliconflow_api_key"
        )
        client = LLMClient(config)
        
        # Use reasoning model for tool calls
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "What tool should I use?"}],
            purpose=ModelPurpose.REASONING
        )
        
        # Use code model for code generation
        code = await client.generate_code(
            prompt="Write a function to calculate mean and std dev",
            language="python"
        )
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client with configuration"""
        self.config = config or LLMConfig.from_env()
        self.logger = logging.getLogger(f"LLMClient.{self.config.provider.value}")
        
        # Validate configuration
        self._validate_config()
        
        # Initialize client based on provider
        self.client = None
        self._init_client()
        
        # Log model configuration
        self.logger.info(
            f"LLM Client initialized: {self.config.provider.value}"
        )
        self.logger.info(f"  Default model: {self.config.model}")
        if self.config.reasoning_model:
            self.logger.info(f"  Reasoning model: {self.config.reasoning_model}")
        if self.config.code_model:
            self.logger.info(f"  Code model: {self.config.code_model}")
        if self.config.writing_model:
            self.logger.info(f"  Writing model: {self.config.writing_model}")
    
    def _validate_config(self):
        """Validate configuration"""
        if not self.config.api_key:
            raise ValueError(
                f"API key required for {self.config.provider.value}. "
                f"Set QUARA_API_KEY or SILICONFLOW_API_KEY environment variable, "
                f"or pass api_key in config."
            )
    
    def _init_client(self):
        """Initialize provider-specific client"""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package required for LLM functionality. "
                "Install with: pip install openai"
            )
        
        if self.config.provider == LLMProvider.SILICONFLOW:
            # SiliconFlow uses OpenAI-compatible API
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base or "https://api.siliconflow.cn/v1",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            
        elif self.config.provider == LLMProvider.OPENAI:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            
        elif self.config.provider == LLMProvider.CUSTOM:
            # Custom OpenAI-compatible endpoint
            if not self.config.api_base:
                raise ValueError("api_base required for custom provider")
            
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _get_model_for_purpose(self, purpose: Optional[ModelPurpose] = None) -> str:
        """
        Select appropriate model based on purpose
        
        Args:
            purpose: The purpose of the request (reasoning, code, writing, general)
            
        Returns:
            Model name to use
        """
        if purpose == ModelPurpose.REASONING and self.config.reasoning_model:
            return self.config.reasoning_model
        elif purpose == ModelPurpose.CODE and self.config.code_model:
            return self.config.code_model
        elif purpose == ModelPurpose.WRITING and self.config.writing_model:
            return self.config.writing_model
        else:
            return self.config.model
    
    def _get_params_for_purpose(self, purpose: Optional[ModelPurpose] = None) -> Dict[str, Any]:
        """
        Get optimal parameters for purpose
        
        Args:
            purpose: The purpose of the request
            
        Returns:
            Dict with temperature and max_tokens
        """
        if purpose == ModelPurpose.REASONING:
            # Lower temperature for consistent tool selection and reasoning
            return {
                "temperature": float(os.getenv("REASONING_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("REASONING_MAX_TOKENS", "4096"))
            }
        elif purpose == ModelPurpose.CODE:
            # Very low temperature for deterministic code generation
            return {
                "temperature": float(os.getenv("CODE_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("CODE_MAX_TOKENS", "8192"))
            }
        elif purpose == ModelPurpose.WRITING:
            # Higher temperature for creative writing
            return {
                "temperature": float(os.getenv("WRITING_TEMPERATURE", "0.8")),
                "max_tokens": int(os.getenv("WRITING_MAX_TOKENS", "8192"))
            }
        else:
            return {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        purpose: Optional[ModelPurpose] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            purpose: Purpose of the request (reasoning, code, writing, general)
            stream: Enable streaming responses
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response dict with 'content' and metadata
        """
        # Select model based on purpose
        model = self._get_model_for_purpose(purpose)
        
        # Get optimal parameters for purpose
        purpose_params = self._get_params_for_purpose(purpose)
        
        # Override with explicit parameters
        final_temperature = temperature if temperature is not None else purpose_params["temperature"]
        final_max_tokens = max_tokens if max_tokens is not None else purpose_params["max_tokens"]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return response
            
            # Extract response content
            result = {
                "content": response.choices[0].message.content,
                "model": response.model,
                "purpose": purpose.value if purpose else "general",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
            self.logger.debug(
                f"Chat completion ({purpose.value if purpose else 'general'}): "
                f"{result['usage']['total_tokens']} tokens used, model: {model}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {str(e)}")
            raise
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Simple text generation interface
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters
            
        Returns:
            Generated text content
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response["content"]
    
    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate code using the code-specialized model
        
        Args:
            prompt: Description of what code to generate
            language: Programming language (python, r, sql, etc.)
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated code
        """
        default_system = f"You are an expert {language} programmer. Generate clean, efficient, well-documented code."
        
        messages = []
        messages.append({"role": "system", "content": system_prompt or default_system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.chat_completion(
            messages=messages,
            purpose=ModelPurpose.CODE,
            **kwargs
        )
        
        return response["content"]
    
    async def reason_and_act(
        self,
        context: str,
        available_tools: List[str],
        task: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use reasoning model for agent thinking and tool selection
        
        Args:
            context: Current context and state
            available_tools: List of available tool names
            task: Task to accomplish
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Dict with thought process and selected action
        """
        default_system = """You are a research agent using the ReAct framework.
Think through the task step by step, then decide what action to take.
Available tools: {tools}

Respond in this format:
Thought: [your reasoning]
Action: [tool name or FINISH]
Action Input: [input for the tool]"""
        
        prompt = f"""Context: {context}

Task: {task}

What should I do next?"""
        
        messages = [
            {"role": "system", "content": (system_prompt or default_system).format(tools=", ".join(available_tools))},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.chat_completion(
            messages=messages,
            purpose=ModelPurpose.REASONING,
            **kwargs
        )
        
        # Parse ReAct format response
        content = response["content"]
        result = {
            "thought": "",
            "action": "",
            "action_input": "",
            "raw_response": content
        }
        
        for line in content.split("\n"):
            if line.startswith("Thought:"):
                result["thought"] = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                result["action"] = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                result["action_input"] = line.replace("Action Input:", "").strip()
        
        return result
    
    async def write_content(
        self,
        prompt: str,
        content_type: str = "academic",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate written content using the writing model
        
        Args:
            prompt: What to write about
            content_type: Type of content (academic, documentation, report, etc.)
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated written content
        """
        default_system = f"You are an expert {content_type} writer. Generate clear, well-structured, professional content."
        
        messages = []
        messages.append({"role": "system", "content": system_prompt or default_system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.chat_completion(
            messages=messages,
            purpose=ModelPurpose.WRITING,
            **kwargs
        )
        
        return response["content"]
    
    async def structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output
        
        Args:
            prompt: User prompt
            schema: JSON schema for output structure
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON object
        """
        # Add JSON formatting instruction to prompt
        json_prompt = f"""{prompt}

Please respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no additional text."""
        
        content = await self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for structured output
            **kwargs
        )
        
        # Extract JSON from response
        try:
            # Try to find JSON in code blocks
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON output: {content}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "api_base": self.config.api_base,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "reasoning_model": self.config.reasoning_model,
            "code_model": self.config.code_model,
            "writing_model": self.config.writing_model
        }
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
        
        # Reinitialize client if provider-related config changed
        if any(k in kwargs for k in ["provider", "api_key", "api_base"]):
            self._init_client()


# Convenience function for quick setup
def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Quick LLM client creation
    
    If no parameters are provided, reads configuration from environment variables.
    
    Example:
        # Using environment variables (.env file)
        client = create_llm_client()  # Reads from env vars
        
        # Explicit configuration
        client = create_llm_client(
            provider="siliconflow",
            model="deepseek-ai/DeepSeek-V3.2-Exp",
            api_key="your_api_key"
        )
        
        # With specialized models
        client = create_llm_client(
            reasoning_model="deepseek-ai/DeepSeek-V3.2-Exp",
            code_model="Qwen/Qwen2.5-Coder-32B-Instruct"
        )
    """
    # If no explicit config provided, load from environment
    if provider is None and model is None and api_key is None and not kwargs:
        config = LLMConfig.from_env()
    else:
        # Explicit configuration with defaults
        config = LLMConfig(
            provider=LLMProvider((provider or "siliconflow").lower()),
            model=model or "deepseek-ai/DeepSeek-V3.2-Exp",
            api_key=api_key or os.getenv("SILICONFLOW_API_KEY") or os.getenv("QUARA_API_KEY"),
            **kwargs
        )
    
    return LLMClient(config)


__all__ = [
    "LLMClient",
    "LLMConfig", 
    "LLMProvider",
    "ModelPurpose",
    "create_llm_client"
]
