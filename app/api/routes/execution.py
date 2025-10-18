from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
import logging

from app.services.execution_service import ExecutionService
from app.models.execution import ExecutionRequest, ExecutionResponse, StreamResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
def get_execution_service() -> ExecutionService:
    return ExecutionService()


@router.post("/", response_model=ExecutionResponse)
async def execute_agent(
    request: ExecutionRequest,
    execution_service: ExecutionService = Depends(get_execution_service)
):
    """Execute agent (non-streaming)"""
    try:
        response = await execution_service.execute_agent(request)
        return response
    except Exception as e:
        logger.error(f"Failed to execute agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute agent"
        )


@router.post("/stream")
async def execute_agent_stream(
    request: ExecutionRequest,
    execution_service: ExecutionService = Depends(get_execution_service)
):
    """Execute agent with streaming response"""
    try:
        # Set stream to True for the request
        request.stream = True
        
        async def generate_stream():
            async for chunk in execution_service.execute_agent_stream(request):
                yield f"data: {chunk.json()}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
    except Exception as e:
        logger.error(f"Failed to execute agent stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute agent stream"
        )


@router.get("/{execution_id}", response_model=ExecutionResponse)
async def get_execution_result(
    execution_id: str,
    execution_service: ExecutionService = Depends(get_execution_service)
):
    """Get execution result by ID"""
    result = await execution_service.get_execution_result(execution_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )
    return result


@router.get("/{execution_id}/trace", response_model=dict)
async def get_execution_trace(
    execution_id: str,
    execution_service: ExecutionService = Depends(get_execution_service)
):
    """Get execution trace for debugging"""
    trace = await execution_service.get_execution_trace(execution_id)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution trace not found"
        )
    return trace


@router.post("/{agent_id}/compile", response_model=dict)
async def compile_agent_graph(
    agent_id: str,
    force_recompile: bool = False,
    execution_service: ExecutionService = Depends(get_execution_service)
):
    """Compile agent graph"""
    try:
        result = await execution_service.compile_agent_graph(agent_id, force_recompile)
        return result
    except Exception as e:
        logger.error(f"Failed to compile agent graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compile agent graph"
        )
