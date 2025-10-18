from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from app.models.agent import AgentConfig


class AgentPublishStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


class PublishedAgent(BaseModel):
    """Published agent for execution - separate from draft"""
    id: str = Field(..., alias="_id")
    original_agent_id: str
    user_id: str
    name: str
    description: str
    version: int
    status: AgentPublishStatus = AgentPublishStatus.PUBLISHED
    config: AgentConfig
    published_at: datetime
    published_by: str
    execution_count: int = 0
    last_executed_at: Optional[datetime] = None
    
    # Execution metadata
    compiled_graph_id: Optional[str] = None
    compilation_time: Optional[float] = None
    dependencies_validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class PublishRequest(BaseModel):
    """Request to publish an agent"""
    agent_id: str
    publish_as_draft: bool = False
    force_recompile: bool = False
    validation_options: Dict[str, Any] = Field(default_factory=dict)


class PublishResponse(BaseModel):
    """Response from agent publish"""
    success: bool
    published_agent_id: Optional[str] = None
    compilation_time: Optional[float] = None
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class AgentPublishHistory(BaseModel):
    """History of agent publications"""
    id: str = Field(..., alias="_id")
    agent_id: str
    published_agent_id: str
    version: int
    published_at: datetime
    published_by: str
    compilation_time: float
    validation_results: Dict[str, Any]
    status: AgentPublishStatus
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class AgentExecutionStats(BaseModel):
    """Statistics for published agent execution"""
    published_agent_id: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: Optional[float] = None
    last_execution_at: Optional[datetime] = None
    execution_errors: Dict[str, int] = Field(default_factory=dict)  # Error type -> count


class AgentPublishValidation(BaseModel):
    """Agent publish validation results"""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    dependencies_validated: bool = False
    graph_compilation_ready: bool = False
    estimated_compilation_time: Optional[float] = None
