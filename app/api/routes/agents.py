from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
import logging

from app.services.agent_service import AgentService
from app.models.agent import (
    Agent, AgentCreateRequest, AgentUpdateRequest, 
    AgentDependency, AgentVersion
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
def get_agent_service() -> AgentService:
    return AgentService()


@router.post("/", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: AgentCreateRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Create a new agent with default configuration"""
    try:
        agent = await agent_service.create_agent(request)
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent"
        )


@router.get("/", response_model=List[Agent])
async def get_agents(
    user_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get all agents for a user"""
    try:
        agents = await agent_service.get_agents_by_user(user_id)
        return agents
    except Exception as e:
        logger.error(f"Failed to get agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agents"
        )


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get agent by ID"""
    agent = await agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent


@router.put("/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Update agent configuration"""
    agent = await agent_service.update_agent(agent_id, request)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent


@router.post("/{agent_id}/publish", response_model=Agent)
async def publish_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Publish agent (mark as published)"""
    # Validate dependencies before publishing
    validation = await agent_service.validate_agent_dependencies(agent_id)
    if not validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent dependencies validation failed: {validation['errors']}"
        )
    
    agent = await agent_service.publish_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent


@router.post("/{agent_id}/archive", response_model=Agent)
async def archive_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Archive agent"""
    agent = await agent_service.archive_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Delete agent"""
    success = await agent_service.delete_agent(agent_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )


@router.get("/{agent_id}/versions", response_model=List[AgentVersion])
async def get_agent_versions(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get all versions for an agent"""
    versions = await agent_service.get_agent_versions(agent_id)
    return versions


@router.get("/{agent_id}/dependencies", response_model=List[AgentDependency])
async def get_agent_dependencies(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get all dependencies for an agent"""
    dependencies = await agent_service.get_agent_dependencies(agent_id)
    return dependencies


@router.post("/{agent_id}/validate", response_model=dict)
async def validate_agent_dependencies(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Validate agent dependencies"""
    validation = await agent_service.validate_agent_dependencies(agent_id)
    return validation


# Component management endpoints

@router.post("/{agent_id}/knowledge-bases", response_model=Agent)
async def add_knowledge_base(
    agent_id: str,
    kb_id: str,
    priority: int = 1,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Add knowledge base to agent"""
    agent = await agent_service.add_knowledge_base(agent_id, kb_id, priority)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent


@router.post("/{agent_id}/tools", response_model=Agent)
async def add_tool(
    agent_id: str,
    tool_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Add tool to agent"""
    agent = await agent_service.add_tool(agent_id, tool_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent


@router.post("/{agent_id}/sub-agents", response_model=Agent)
async def add_sub_agent(
    agent_id: str,
    sub_agent_id: str,
    trigger_condition: Optional[str] = None,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Add sub-agent to agent"""
    agent = await agent_service.add_sub_agent(agent_id, sub_agent_id, trigger_condition)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent


@router.delete("/{agent_id}/knowledge-bases/{kb_id}", response_model=Agent)
async def remove_knowledge_base(
    agent_id: str,
    kb_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Remove knowledge base from agent"""
    agent = await agent_service.remove_knowledge_base(agent_id, kb_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or knowledge base not found"
        )
    return agent


@router.delete("/{agent_id}/tools/{tool_id}", response_model=Agent)
async def remove_tool(
    agent_id: str,
    tool_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Remove tool from agent"""
    agent = await agent_service.remove_tool(agent_id, tool_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or tool not found"
        )
    return agent


@router.delete("/{agent_id}/sub-agents/{sub_agent_id}", response_model=Agent)
async def remove_sub_agent(
    agent_id: str,
    sub_agent_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Remove sub-agent from agent"""
    agent = await agent_service.remove_sub_agent(agent_id, sub_agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or sub-agent not found"
        )
    return agent


@router.patch("/{agent_id}/knowledge-bases/{kb_id}/toggle", response_model=Agent)
async def toggle_knowledge_base(
    agent_id: str,
    kb_id: str,
    enabled: bool,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Toggle specific knowledge base in agent"""
    agent = await agent_service.toggle_knowledge_base(agent_id, kb_id, enabled)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or knowledge base not found"
        )
    return agent


@router.patch("/{agent_id}/tools/{tool_id}/toggle", response_model=Agent)
async def toggle_tool(
    agent_id: str,
    tool_id: str,
    enabled: bool,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Toggle specific tool in agent"""
    agent = await agent_service.toggle_tool(agent_id, tool_id, enabled)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or tool not found"
        )
    return agent


@router.patch("/{agent_id}/sub-agents/{sub_agent_id}/toggle", response_model=Agent)
async def toggle_sub_agent(
    agent_id: str,
    sub_agent_id: str,
    enabled: bool,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Toggle specific sub-agent in agent"""
    agent = await agent_service.toggle_sub_agent(agent_id, sub_agent_id, enabled)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or sub-agent not found"
        )
    return agent


@router.put("/{agent_id}/knowledge-bases/{kb_id}/config", response_model=Agent)
async def update_knowledge_base_config(
    agent_id: str,
    kb_id: str,
    config: dict,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Update knowledge base configuration in agent"""
    agent = await agent_service.update_knowledge_base_config(agent_id, kb_id, config)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or knowledge base not found"
        )
    return agent


@router.put("/{agent_id}/tools/{tool_id}/config", response_model=Agent)
async def update_tool_config(
    agent_id: str,
    tool_id: str,
    config: dict,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Update tool configuration in agent"""
    agent = await agent_service.update_tool_config(agent_id, tool_id, config)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or tool not found"
        )
    return agent


@router.put("/{agent_id}/sub-agents/{sub_agent_id}/config", response_model=Agent)
async def update_sub_agent_config(
    agent_id: str,
    sub_agent_id: str,
    trigger_condition: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Update sub-agent configuration in agent"""
    agent = await agent_service.update_sub_agent_config(agent_id, sub_agent_id, trigger_condition)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent or sub-agent not found"
        )
    return agent
