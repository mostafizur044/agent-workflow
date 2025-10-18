from pydantic import BaseModel, Field, SecretStr
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ModelProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class ModelType(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE = "image"


class ModelConfig(BaseModel):
    """Base model configuration"""
    provider: ModelProvider
    model_name: str
    model_type: ModelType
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    custom_headers: Optional[Dict[str, str]] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None


class OpenAIConfig(ModelConfig):
    """OpenAI specific configuration"""
    provider: Literal[ModelProvider.OPENAI] = ModelProvider.OPENAI
    organization: Optional[str] = None


class AzureConfig(ModelConfig):
    """Azure OpenAI specific configuration"""
    provider: Literal[ModelProvider.AZURE] = ModelProvider.AZURE
    deployment_name: str
    api_version: str = "2024-02-15-preview"
    azure_endpoint: str
    azure_ad_token: Optional[SecretStr] = None


class AnthropicConfig(ModelConfig):
    """Anthropic specific configuration"""
    provider: Literal[ModelProvider.ANTHROPIC] = ModelProvider.ANTHROPIC
    anthropic_version: str = "2023-06-01"


class GoogleConfig(ModelConfig):
    """Google specific configuration"""
    provider: Literal[ModelProvider.GOOGLE] = ModelProvider.GOOGLE
    project_id: Optional[str] = None
    location: Optional[str] = None


class CohereConfig(ModelConfig):
    """Cohere specific configuration"""
    provider: Literal[ModelProvider.COHERE] = ModelProvider.COHERE


class OpenRouterConfig(ModelConfig):
    """OpenRouter specific configuration"""
    provider: Literal[ModelProvider.OPENROUTER] = ModelProvider.OPENROUTER
    app_name: Optional[str] = None
    website: Optional[str] = None


class LocalConfig(ModelConfig):
    """Local model configuration"""
    provider: Literal[ModelProvider.LOCAL] = ModelProvider.LOCAL
    local_path: str
    device: Optional[str] = None
    gpu_layers: Optional[int] = None


class Model(BaseModel):
    """Model registry entry"""
    id: str = Field(..., alias="_id")
    user_id: str
    name: str
    description: str
    provider: ModelProvider
    model_name: str
    model_type: ModelType
    config: Dict[str, Any]
    is_active: bool = True
    is_shared: bool = False
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        use_enum_values = True
        populate_by_name = True


class ModelCreateRequest(BaseModel):
    """Request to create a new model"""
    name: str
    description: str
    provider: ModelProvider
    model_name: str
    model_type: ModelType
    config: Dict[str, Any]
    is_shared: bool = False
    tags: List[str] = []
    user_id: str


class ModelUpdateRequest(BaseModel):
    """Request to update a model"""
    name: Optional[str] = None
    description: Optional[str] = None
    model_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    is_shared: Optional[bool] = None
    tags: Optional[List[str]] = None


class ModelTestRequest(BaseModel):
    """Request to test a model"""
    test_prompt: str = "Hello, how are you?"
    max_tokens: Optional[int] = 100


class ModelTestResponse(BaseModel):
    """Response from model test"""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    execution_time: float
    tokens_used: Optional[int] = None


class ModelCapabilities(BaseModel):
    """Model capabilities information"""
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    max_context_length: Optional[int] = None
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None


class PresetModel(BaseModel):
    """Preset model configurations"""
    name: str
    provider: ModelProvider
    model_name: str
    model_type: ModelType
    config: Dict[str, Any]
    capabilities: ModelCapabilities
    description: str


# Preset models for easy setup
PRESET_MODELS = [
    PresetModel(
        name="GPT-4 Turbo",
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-turbo-preview",
        model_type=ModelType.CHAT,
        config={
            "max_tokens": 4096,
            "temperature": 0.7,
            "supports_function_calling": True,
            "supports_vision": True
        },
        capabilities=ModelCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            supports_json_mode=True,
            max_context_length=128000
        ),
        description="Latest GPT-4 model with vision and function calling support"
    ),
    PresetModel(
        name="GPT-3.5 Turbo",
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        model_type=ModelType.CHAT,
        config={
            "max_tokens": 4096,
            "temperature": 0.7
        },
        capabilities=ModelCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            max_context_length=16385
        ),
        description="Fast and cost-effective GPT-3.5 model"
    ),
    PresetModel(
        name="Claude 3 Opus",
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        model_type=ModelType.CHAT,
        config={
            "max_tokens": 4096,
            "temperature": 0.7
        },
        capabilities=ModelCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            max_context_length=200000
        ),
        description="Anthropic's most capable model"
    ),
    PresetModel(
        name="Claude 3 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        model_type=ModelType.CHAT,
        config={
            "max_tokens": 4096,
            "temperature": 0.7
        },
        capabilities=ModelCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            max_context_length=200000
        ),
        description="Balanced performance and cost Claude model"
    ),
    PresetModel(
        name="Gemini Pro",
        provider=ModelProvider.GOOGLE,
        model_name="gemini-pro",
        model_type=ModelType.CHAT,
        config={
            "max_tokens": 8192,
            "temperature": 0.7
        },
        capabilities=ModelCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            max_context_length=32768
        ),
        description="Google's advanced language model"
    ),
    PresetModel(
        name="Text Embedding Ada 002",
        provider=ModelProvider.OPENAI,
        model_name="text-embedding-ada-002",
        model_type=ModelType.EMBEDDING,
        config={
            "dimensions": 1536
        },
        capabilities=ModelCapabilities(
            supports_streaming=False,
            supports_function_calling=False,
            max_context_length=8191
        ),
        description="OpenAI's efficient embedding model"
    ),
    PresetModel(
        name="Text Embedding 3 Small",
        provider=ModelProvider.OPENAI,
        model_name="text-embedding-3-small",
        model_type=ModelType.EMBEDDING,
        config={
            "dimensions": 1536
        },
        capabilities=ModelCapabilities(
            supports_streaming=False,
            supports_function_calling=False,
            max_context_length=8191
        ),
        description="OpenAI's latest small embedding model"
    ),
    PresetModel(
        name="Text Embedding 3 Large",
        provider=ModelProvider.OPENAI,
        model_name="text-embedding-3-large",
        model_type=ModelType.EMBEDDING,
        config={
            "dimensions": 3072
        },
        capabilities=ModelCapabilities(
            supports_streaming=False,
            supports_function_calling=False,
            max_context_length=8191
        ),
        description="OpenAI's latest large embedding model"
    )
]
