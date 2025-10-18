from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
import logging

from app.services.agent_publish_service import AgentPublishService
from app.models.agent_publish import (
    PublishRequest, PublishResponse, AgentPublishValidation,
    AgentPublishHistory, AgentExecutionStats, PublishedAgent
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
def get_publish_service() -> AgentPublishService:
    return AgentPublishService()


@router.post("/validate/{agent_id}", response_model=AgentPublishValidation)
async def validate_agent_for_publish(
    agent_id: str,
    publish_service: AgentPublishService = Depends(get_publish_service)
):
    """Validate agent configuration before publishing"""
    try:
        validation = await publish_service.validate_agent_for_publish(agent_id)
        return validation
    except Exception as e:
        logger.error(f"Failed to validate agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate agent"
        )


@router.post("/publish", response_model=PublishResponse)
async def publish_agent(
    request: PublishRequest,
    publish_service: AgentPublishService = Depends(get_publish_service)
):
    """Publish an agent for execution"""
    try:
        response = await publish_service.publish_agent(request)
        return response
    except Exception as e:
        logger.error(f"Failed to publish agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to publish agent"
        )


@router.get("/published/{published_agent_id}", response_model=PublishedAgent)
async def get_published_agent(
    published_agent_id: str,
    publish_service: AgentPublishService = Depends(get_publish_service)
):
    """Get published agent by ID"""
    agent = await publish_service.get_published_agent(published_agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Published agent not found"
        )
    return agent


@router.get("/published/user/{user_id}", response_model=List[PublishedAgent])
async def get_published_agents_by_user(
    user_id: str,
    publish_service: AgentPublishService = Depends(get_publish_service)
):
    """Get all published agents for a user"""
    try:
        agents = await publish_service.get_published_agents_by_user(user_id)
        return agents
    except Exception as e:
        logger.error(f"Failed to get published agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve published agents"
        )


@router.get("/published/agent/{original_agent_id}", response_model=List[PublishedAgent])
async def get_published_agents_by_original(
    original_agent_id: str,
    publish_service: AgentPublishService = Depends(get_publish_service)
):
    """Get all published versions of an agent"""
    try:
        agents = await publish_service.get_published_agents_by_original(original_agent_id)
        return agents
    except Exception as e:
        logger.error(f"Failed to get published agent versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve published agent versions"
        )


@router.delete("/published/{published_agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unpublish_agent(
    published_agent_id: str,
    publish_service: AgentPublishService = Depends(get_publish_service)
):
    """Unpublish an agent"""
    success = await publish_service.unpublish_agent(published_agent_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Published agent not found"
        )


@router.get("/history/{agent_id}", response_model=List[AgentPublishHistory])
async def get_publish_history(
    agent_id: str,
    publish_service: AgentPublishService = Depends(get_publish_service)
):
    """Get publish history for an agent"""
    try:
        history = await publish_service.get_publish_history(agent_id)
        return history
    except Exception as e:
        logger.error(f"Failed to get publish history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve publish history"
        )


@router.get("/stats/{published_agent_id}", response_model=AgentExecutionStats)
async def get_execution_stats(
    published_agent_id: str,
    publish_service: AgentPublishService = Depends(get_publish_service)
):
    """Get execution statistics for a published agent"""
    try:
        stats = await publish_service.get_execution_stats(published_agent_id)
        return stats
    except Exception as e:
        logger.error(f"Failed to get execution stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution statistics"
        )
