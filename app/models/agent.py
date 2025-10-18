from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class AgentStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class LLMConfig(BaseModel):
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = "You are a helpful assistant."
    streaming: bool = True


class KnowledgeBaseItem(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    enabled: bool = True
    priority: int = 1
    retrieval_config: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class KnowledgeBasesConfig(BaseModel):
    enabled: bool = False
    items: List[KnowledgeBaseItem] = []


class ToolItem(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    type: str
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class ToolsConfig(BaseModel):
    enabled: bool = False
    items: List[ToolItem] = []


class MCPServerItem(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    enabled: bool = False
    connection: Dict[str, Any]
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class MCPServersConfig(BaseModel):
    enabled: bool = False
    items: List[MCPServerItem] = []


class SubAgentItem(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    enabled: bool = False
    trigger_condition: Optional[str] = None
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class SubAgentsConfig(BaseModel):
    enabled: bool = False
    items: List[SubAgentItem] = []


class MemoryConfig(BaseModel):
    enabled: bool = True
    type: str = "buffer"
    window_size: int = 10
    summarization: bool = True


class RoutingConfig(BaseModel):
    type: str = "conditional"  # or "sequential"
    rules: List[Dict[str, Any]] = []


class AgentMetadata(BaseModel):
    created_at: datetime
    updated_at: datetime
    created_by: str
    published_at: Optional[datetime] = None


class AgentConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    knowledge_bases: KnowledgeBasesConfig = Field(default_factory=KnowledgeBasesConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    mcp_servers: MCPServersConfig = Field(default_factory=MCPServersConfig)
    sub_agents: SubAgentsConfig = Field(default_factory=SubAgentsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    metadata: AgentMetadata


class Agent(BaseModel):
    id: str = Field(..., alias="_id")
    user_id: str
    name: str
    description: str
    version: int = 1
    status: AgentStatus = AgentStatus.DRAFT
    config: AgentConfig
    parent_agent_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None

    class Config:
        use_enum_values = True
        populate_by_name = True


class AgentDependency(BaseModel):
    id: str = Field(..., alias="_id")
    agent_id: str
    dependency_type: Literal["kb", "tool", "mcp", "sub_agent"]
    dependency_id: str
    enabled: bool = True
    config_override: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class AgentVersion(BaseModel):
    id: str = Field(..., alias="_id")
    agent_id: str
    version: int
    config_snapshot: AgentConfig
    created_at: datetime
    
    class Config:
        use_enum_values = True
        populate_by_name = True

class AgentLink(BaseModel):
    id: str = Field(..., alias="_id")
    parent_agent_id: str
    child_agent_id: str
    trigger_type: Literal["conditional", "always", "manual"]
    trigger_condition: Optional[str] = None
    input_mapping: Optional[Dict[str, str]] = None
    output_mapping: Optional[Dict[str, str]] = None
    enabled: bool = True
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class AgentCreateRequest(BaseModel):
    name: str
    description: str
    user_id: str


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[AgentConfig] = None
