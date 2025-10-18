from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ToolType(str, Enum):
    API = "api"
    FUNCTION = "function"
    MCP = "mcp"
    WEBHOOK = "webhook"
    LANGCHAIN = "langchain"
    CUSTOM = "custom"


class ToolStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class APIConfig(BaseModel):
    base_url: str
    headers: Dict[str, str] = {}
    timeout: int = 30
    retries: int = 3


class FunctionConfig(BaseModel):
    code: str
    dependencies: List[str] = []
    environment_variables: Dict[str, str] = {}


class MCPConfig(BaseModel):
    server_url: str
    capabilities: List[str] = []
    authentication: Optional[Dict[str, Any]] = None


class WebhookConfig(BaseModel):
    url: str
    method: str = "POST"
    headers: Dict[str, str] = {}
    payload_template: Optional[str] = None


class LangChainConfig(BaseModel):
    tool_name: str
    tool_class: str
    parameters: Dict[str, Any] = {}
    import_path: Optional[str] = None


class CustomConfig(BaseModel):
    implementation: str  # Base64 encoded or file reference
    language: str = "python"
    dependencies: List[str] = []
    environment_variables: Dict[str, str] = {}


class ToolConfig(BaseModel):
    api: Optional[APIConfig] = None
    function: Optional[FunctionConfig] = None
    mcp: Optional[MCPConfig] = None
    webhook: Optional[WebhookConfig] = None
    langchain: Optional[LangChainConfig] = None
    custom: Optional[CustomConfig] = None


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class Tool(BaseModel):
    id: str = Field(..., alias="_id")
    user_id: str
    name: str
    description: str
    type: ToolType
    status: ToolStatus = ToolStatus.ACTIVE
    config: ToolConfig
    parameters: List[ToolParameter] = []
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        use_enum_values = True
        populate_by_name = True


class ToolCreateRequest(BaseModel):
    name: str
    description: str
    type: ToolType
    config: ToolConfig
    parameters: List[ToolParameter] = []
    tags: List[str] = []
    user_id: str


class ToolUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[ToolType] = None
    config: Optional[ToolConfig] = None
    parameters: Optional[List[ToolParameter]] = None
    tags: Optional[List[str]] = None


class ToolExecutionRequest(BaseModel):
    tool_id: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class ToolExecutionResponse(BaseModel):
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None
