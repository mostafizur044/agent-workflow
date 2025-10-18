from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
import logging

from app.services.tool_service import ToolService
from app.models.tool import (
    Tool, ToolCreateRequest, ToolUpdateRequest, 
    ToolExecutionRequest, ToolExecutionResponse, ToolType
)
from app.models.langchain_tools import (
    LangChainTool, LangChainToolCategory, LangChainToolListResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
def get_tool_service() -> ToolService:
    return ToolService()


@router.post("/", response_model=Tool, status_code=status.HTTP_201_CREATED)
async def create_tool(
    request: ToolCreateRequest,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Create a new tool"""
    try:
        # Validate tool configuration
        validation = await tool_service.validate_tool_config(request.type, request.config)
        if not validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tool configuration validation failed: {validation['errors']}"
            )
        
        tool = await tool_service.create_tool(request)
        return tool
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create tool: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create tool"
        )


@router.get("/", response_model=List[Tool])
async def get_tools(
    user_id: str,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Get all tools for a user"""
    try:
        tools = await tool_service.get_tools_by_user(user_id)
        return tools
    except Exception as e:
        logger.error(f"Failed to get tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tools"
        )


@router.get("/{tool_id}", response_model=Tool)
async def get_tool(
    tool_id: str,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Get tool by ID"""
    tool = await tool_service.get_tool(tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found"
        )
    return tool


@router.put("/{tool_id}", response_model=Tool)
async def update_tool(
    tool_id: str,
    request: ToolUpdateRequest,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Update tool"""
    tool = await tool_service.update_tool(tool_id, request)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found"
        )
    return tool


@router.delete("/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tool(
    tool_id: str,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Delete tool"""
    success = await tool_service.delete_tool(tool_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found"
        )


@router.post("/{tool_id}/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    tool_id: str,
    request: ToolExecutionRequest,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Execute a tool"""
    try:
        response = await tool_service.execute_tool(request)
        return response
    except Exception as e:
        logger.error(f"Failed to execute tool: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute tool"
        )


@router.post("/validate", response_model=dict)
async def validate_tool_config(
    tool_type: ToolType,
    config: dict,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Validate tool configuration"""
    try:
        validation = await tool_service.validate_tool_config(tool_type, config)
        return validation
    except Exception as e:
        logger.error(f"Failed to validate tool config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate tool configuration"
        )


# LangChain Tools endpoints

@router.get("/langchain/list", response_model=LangChainToolListResponse)
async def get_langchain_tools(
    category: Optional[LangChainToolCategory] = None,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Get available LangChain tools"""
    try:
        tools = await tool_service.get_langchain_tools(category)
        categories = list(LangChainToolCategory)
        
        return LangChainToolListResponse(
            tools=tools,
            categories=categories,
            total_count=len(tools)
        )
    except Exception as e:
        logger.error(f"Failed to get LangChain tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve LangChain tools"
        )


@router.get("/langchain/{tool_name}", response_model=LangChainTool)
async def get_langchain_tool(
    tool_name: str,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Get specific LangChain tool"""
    try:
        tool = await tool_service.get_langchain_tool(tool_name)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LangChain tool not found"
            )
        return tool
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get LangChain tool: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve LangChain tool"
        )


@router.post("/langchain/{tool_name}/create", response_model=Tool)
async def create_langchain_tool(
    tool_name: str,
    user_id: str,
    configuration: Optional[dict] = None,
    tool_service: ToolService = Depends(get_tool_service)
):
    """Create a tool from LangChain tool definition"""
    try:
        tool = await tool_service.create_langchain_tool(tool_name, user_id, configuration or {})
        return tool
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create LangChain tool: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create LangChain tool"
        )
