from typing import Optional, Dict, Any, AsyncGenerator
from uuid import uuid4
from datetime import datetime
import logging
import asyncio

from app.services.agent_service import AgentService
from app.core.graph_compiler import GraphCompiler
from app.models.execution import (
    ExecutionRequest, ExecutionResponse, ExecutionStatus,
    StreamResponse, ExecutionTrace, AgentState
)
from app.core.cache import get_cache_manager

logger = logging.getLogger(__name__)


class ExecutionService:
    """Service for executing agents and managing execution traces"""
    
    def __init__(self):
        self.agent_service = AgentService()
        self.graph_compiler = GraphCompiler()
        self.active_executions: Dict[str, asyncio.Task] = {}
    
    async def execute_agent(self, request: ExecutionRequest) -> ExecutionResponse:
        """Execute agent (non-streaming)"""
        execution_id = str(uuid4())
        start_time = datetime.now()
        
        try:
            # Get agent
            agent = await self.agent_service.get_agent(request.agent_id)
            if not agent:
                return ExecutionResponse(
                    id=execution_id,
                    agent_id=request.agent_id,
                    status=ExecutionStatus.FAILED,
                    input=request.input,
                    error="Agent not found",
                    execution_time=0.0,
                    created_at=start_time
                )
            
            # Check if agent is published
            if agent.status != "published":
                return ExecutionResponse(
                    id=execution_id,
                    agent_id=request.agent_id,
                    status=ExecutionStatus.FAILED,
                    input=request.input,
                    error="Agent is not published",
                    execution_time=0.0,
                    created_at=start_time
                )
            
            # Compile or get cached graph
            graph = await self.graph_compiler.get_or_compile_graph(agent)
            if not graph:
                return ExecutionResponse(
                    id=execution_id,
                    agent_id=request.agent_id,
                    status=ExecutionStatus.FAILED,
                    input=request.input,
                    error="Failed to compile agent graph",
                    execution_time=0.0,
                    created_at=start_time
                )
            
            # Initialize state
            initial_state = AgentState(
                messages=[{"role": "user", "content": request.input}],
                metadata={
                    "execution_id": str(execution_id),
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "context": request.context or {}
                }
            )
            
            # Execute graph
            execution_trace = ExecutionTrace(
                execution_id=execution_id,
                total_execution_time=0.0
            )
            
            result = await graph.ainvoke(initial_state.model_dump())
            
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_trace.total_execution_time = execution_time
            
            # Cache execution trace
            await get_cache_manager().set_execution_trace(str(execution_id), execution_trace.model_dump())
            
            return ExecutionResponse(
                id=execution_id,
                agent_id=request.agent_id,
                status=ExecutionStatus.COMPLETED,
                input=request.input,
                output=result.get("messages", [{}])[-1].get("content", ""),
                context=result.get("metadata", {}),
                execution_trace=execution_trace.model_dump(),
                execution_time=execution_time,
                created_at=start_time,
                completed_at=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Agent execution failed: {e}")
            
            return ExecutionResponse(
                id=execution_id,
                agent_id=request.agent_id,
                status=ExecutionStatus.FAILED,
                input=request.input,
                error=str(e),
                execution_time=execution_time,
                created_at=start_time,
                completed_at=datetime.now()
            )
    
    async def execute_agent_stream(self, request: ExecutionRequest) -> AsyncGenerator[StreamResponse, None]:
        """Execute agent with streaming response"""
        execution_id = str(uuid4())
        start_time = datetime.now()
        
        try:
            yield StreamResponse(
                type="start",
                data={"execution_id": str(execution_id)},
                timestamp=start_time
            )
            
            # Get agent
            agent = await self.agent_service.get_agent(request.agent_id)
            if not agent:
                yield StreamResponse(
                    type="error",
                    data={"error": "Agent not found"},
                    timestamp=datetime.now()
                )
                return
            
            # Check if agent is published
            if agent.status != "published":
                yield StreamResponse(
                    type="error",
                    data={"error": "Agent is not published"},
                    timestamp=datetime.now()
                )
                return
            
            # Compile or get cached graph
            graph = await self.graph_compiler.get_or_compile_graph(agent)
            if not graph:
                yield StreamResponse(
                    type="error",
                    data={"error": "Failed to compile agent graph"},
                    timestamp=datetime.now()
                )
                return
            
            yield StreamResponse(
                type="compiled",
                data={"message": "Graph compiled successfully"},
                timestamp=datetime.now()
            )
            
            # Initialize state
            initial_state = AgentState(
                messages=[{"role": "user", "content": request.input}],
                metadata={
                    "execution_id": str(execution_id),
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "context": request.context or {}
                }
            )
            
            # Stream execution
            async for chunk in graph.astream(initial_state.model_dump()):
                yield StreamResponse(
                    type="step",
                    data=chunk,
                    timestamp=datetime.now()
                )
            
            yield StreamResponse(
                type="complete",
                data={"message": "Execution completed"},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Streaming execution failed: {e}")
            yield StreamResponse(
                type="error",
                data={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def get_execution_result(self, execution_id: str) -> Optional[ExecutionResponse]:
        """Get execution result by ID"""
        # This would typically be stored in a database
        # For now, we'll return None as a placeholder
        return None
    
    async def get_execution_trace(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution trace for debugging"""
        trace = await get_cache_manager().get_execution_trace(str(execution_id))
        return trace
    
    async def compile_agent_graph(self, agent_id: str, force_recompile: bool = False) -> Dict[str, Any]:
        """Compile agent graph"""
        agent = await self.agent_service.get_agent(agent_id)
        if not agent:
            raise ValueError("Agent not found")
        
        start_time = datetime.now()
        
        try:
            # Check if already compiled and cached
            if not force_recompile:
                cached_graph = await get_cache_manager().get_agent_graph(str(agent_id), agent.version)
                if cached_graph:
                    return {
                        "success": True,
                        "compilation_time": 0.0,
                        "nodes_created": [],
                        "edges_created": [],
                        "cached": True
                    }
            
            # Compile graph
            graph = await self.graph_compiler.compile_agent_graph(agent)
            
            compilation_time = (datetime.now() - start_time).total_seconds()
            
            # Cache the compiled graph
            await get_cache_manager().set_agent_graph(str(agent_id), agent.version, graph)
            
            return {
                "success": True,
                "compilation_time": compilation_time,
                "nodes_created": list(graph.nodes.keys()) if hasattr(graph, 'nodes') else [],
                "edges_created": list(graph.edges) if hasattr(graph, 'edges') else [],
                "cached": False
            }
            
        except Exception as e:
            compilation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Graph compilation failed: {e}")
            
            return {
                "success": False,
                "compilation_time": compilation_time,
                "nodes_created": [],
                "edges_created": [],
                "cached": False,
                "error": str(e)
            }
