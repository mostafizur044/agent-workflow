from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionRequest(BaseModel):
    agent_id: str
    input: str
    context: Optional[Dict[str, Any]] = None
    stream: bool = False
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ExecutionResponse(BaseModel):
    id: str = Field(..., alias="_id")
    agent_id: str
    status: ExecutionStatus
    input: str
    output: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    execution_trace: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True
        populate_by_name = True


class ExecutionTrace(BaseModel):
    execution_id: str
    steps: List['ExecutionStep'] = []
    total_execution_time: float
    memory_usage: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None


class ExecutionStep(BaseModel):
    step_id: str
    node_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    error: Optional[str] = None


# Update forward references
ExecutionTrace.model_rebuild()


class StreamResponse(BaseModel):
    type: str  # "chunk", "step", "error", "complete"
    data: Any
    timestamp: datetime


class AgentState(BaseModel):
    messages: List[Dict[str, Any]] = []
    current_step: str = "entry"
    retrieved_context: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    sub_agent_responses: List[Dict[str, Any]] = []
    conversation_history: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphCompilationRequest(BaseModel):
    agent_id: str
    force_recompile: bool = False


class GraphCompilationResponse(BaseModel):
    success: bool
    compilation_time: float
    nodes_created: List[str]
    edges_created: List[tuple]
    cached: bool
    error: Optional[str] = None
