from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

from app.database import get_database
from app.services.agent_service import AgentService
from app.services.model_service import ModelService
from app.core.graph_compiler import GraphCompiler
from app.models.agent_publish import (
    PublishedAgent, PublishRequest, PublishResponse, AgentPublishStatus,
    AgentPublishHistory, AgentExecutionStats, AgentPublishValidation
)
from app.core.cache import get_cache_manager

logger = logging.getLogger(__name__)


class AgentPublishService:
    """Service for managing agent publishing and execution"""
    
    def __init__(self):
        self.db: AsyncIOMotorDatabase = get_database()
        self.agent_service = AgentService()
        self.model_service = ModelService()
        self.graph_compiler = GraphCompiler()
    
    async def validate_agent_for_publish(self, agent_id: str) -> AgentPublishValidation:
        """Validate agent configuration before publishing"""
        agent = await self.agent_service.get_agent(agent_id)
        if not agent:
            return AgentPublishValidation(
                valid=False,
                errors=["Agent not found"]
            )
        
        errors = []
        warnings = []
        
        # Validate LLM configuration
        if agent.config.llm.enabled:
            if not agent.config.llm.model:
                errors.append("LLM model not specified")
            if not agent.config.llm.system_prompt:
                warnings.append("No system prompt specified")
        
        # Validate knowledge bases
        if agent.config.knowledge_bases.enabled:
            for kb_item in agent.config.knowledge_bases.items:
                if kb_item.enabled:
                    kb = await self.db.get_collection("knowledge_bases").find_one({
                        "_id": str(kb_item.id),
                        "status": "active"
                    })
                    if not kb:
                        errors.append(f"Knowledge base {kb_item.name} not found or inactive")
        
        # Validate tools
        if agent.config.tools.enabled:
            for tool_item in agent.config.tools.items:
                if tool_item.enabled:
                    tool = await self.db.get_collection("tools").find_one({
                        "_id": str(tool_item.id),
                        "status": "active"
                    })
                    if not tool:
                        errors.append(f"Tool {tool_item.name} not found or inactive")
        
        # Validate sub-agents
        if agent.config.sub_agents.enabled:
            for sub_agent_item in agent.config.sub_agents.items:
                if sub_agent_item.enabled:
                    # Check if sub-agent is published
                    published_sub_agent = await self.db.get_collection("published_agents").find_one({
                        "original_agent_id": str(sub_agent_item.id),
                        "status": "published"
                    })
                    if not published_sub_agent:
                        errors.append(f"Sub-agent {sub_agent_item.name} is not published")
        
        # Validate memory configuration
        if agent.config.memory.enabled:
            if agent.config.memory.window_size <= 0:
                errors.append("Memory window size must be positive")
        
        return AgentPublishValidation(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            dependencies_validated=len(errors) == 0,
            graph_compilation_ready=len(errors) == 0
        )
    
    async def publish_agent(self, request: PublishRequest) -> PublishResponse:
        """Publish an agent for execution"""
        start_time = datetime.now()
        
        try:
            # Get the draft agent
            draft_agent = await self.agent_service.get_agent(request.agent_id)
            if not draft_agent:
                return PublishResponse(
                    success=False,
                    errors=["Agent not found"]
                )
            
            # Validate agent for publishing
            validation = await self.validate_agent_for_publish(request.agent_id)
            if not validation.valid:
                return PublishResponse(
                    success=False,
                    errors=validation.errors,
                    warnings=validation.warnings,
                    validation_results=validation.model_dump()
                )
            
            # Create published agent
            published_agent_id = str(uuid4())
            published_at = datetime.now()
            
            published_agent = PublishedAgent(
                id=published_agent_id,
                original_agent_id=request.agent_id,
                user_id=draft_agent.user_id,
                name=draft_agent.name,
                description=draft_agent.description,
                version=draft_agent.version,
                status=AgentPublishStatus.PUBLISHED,
                config=draft_agent.config,
                published_at=published_at,
                published_by=draft_agent.user_id,  # In real app, this would be current user
                dependencies_validated=True
            )
            
            # Compile the graph
            compilation_start = datetime.now()
            try:
                compiled_graph = await self.graph_compiler.compile_agent_graph(draft_agent)
                compilation_time = (datetime.now() - compilation_start).total_seconds()
                
                # Cache the compiled graph
                await get_cache_manager().set_agent_graph(str(published_agent_id), draft_agent.version, compiled_graph)
                
                published_agent.compiled_graph_id = str(published_agent_id)
                published_agent.compilation_time = compilation_time
                
            except Exception as e:
                logger.error(f"Graph compilation failed: {e}")
                return PublishResponse(
                    success=False,
                    errors=[f"Graph compilation failed: {str(e)}"],
                    validation_results=validation.model_dump()
                )
            
            # Save published agent to separate collection
            await self.db.get_collection("published_agents").insert_one(published_agent.model_dump(by_alias=True))
            
            # Create publish history entry
            history_entry = AgentPublishHistory(
                id=str(uuid4()),
                agent_id=request.agent_id,
                published_agent_id=published_agent_id,
                version=draft_agent.version,
                published_at=published_at,
                published_by=draft_agent.user_id,
                compilation_time=compilation_time,
                validation_results=validation.model_dump(),
                status=AgentPublishStatus.PUBLISHED
            )
            
            await self.db.get_collection("agent_publish_history").insert_one(history_entry.model_dump(by_alias=True))
            
            # Update draft agent status
            await self.agent_service.publish_agent(request.agent_id)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return PublishResponse(
                success=True,
                published_agent_id=published_agent_id,
                compilation_time=compilation_time,
                validation_results=validation.model_dump(),
                warnings=validation.warnings
            )
            
        except Exception as e:
            logger.error(f"Agent publishing failed: {e}")
            return PublishResponse(
                success=False,
                errors=[f"Publishing failed: {str(e)}"]
            )
    
    async def get_published_agent(self, published_agent_id: str) -> Optional[PublishedAgent]:
        """Get published agent by ID"""
        doc = await self.db.get_collection("published_agents").find_one({"_id": str(published_agent_id)})
        if doc:
            doc["id"] = doc.pop("_id")
            return PublishedAgent(**doc)
        return None
    
    async def get_published_agents_by_user(self, user_id: str) -> List[PublishedAgent]:
        """Get all published agents for a user"""
        cursor = self.db.get_collection("published_agents").find({"user_id": user_id}).sort("published_at", -1)
        agents = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            agents.append(PublishedAgent(**doc))
        return agents
    
    async def get_published_agents_by_original(self, original_agent_id: str) -> List[PublishedAgent]:
        """Get all published versions of an agent"""
        cursor = self.db.get_collection("published_agents").find({"original_agent_id": str(original_agent_id)}).sort("version", -1)
        agents = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            agents.append(PublishedAgent(**doc))
        return agents
    
    async def unpublish_agent(self, published_agent_id: str) -> bool:
        """Unpublish an agent (mark as inactive)"""
        result = await self.db.get_collection("published_agents").update_one(
            {"_id": str(published_agent_id)},
            {
                "$set": {
                    "status": AgentPublishStatus.FAILED,
                    "unpublished_at": datetime.now()
                }
            }
        )
        
        if result.modified_count > 0:
            # Invalidate cache
            await get_cache_manager().invalidate_agent_cache(str(published_agent_id))
            return True
        
        return False
    
    async def get_publish_history(self, agent_id: str) -> List[AgentPublishHistory]:
        """Get publish history for an agent"""
        cursor = self.db.get_collection("agent_publish_history").find({"agent_id": str(agent_id)}).sort("published_at", -1)
        history = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            history.append(AgentPublishHistory(**doc))
        return history
    
    async def get_execution_stats(self, published_agent_id: str) -> Optional[AgentExecutionStats]:
        """Get execution statistics for a published agent"""
        doc = await self.db.get_collection("agent_execution_stats").find_one({"_id": str(published_agent_id)})
        if doc:
            doc["published_agent_id"] = doc.pop("_id")
            return AgentExecutionStats(**doc)
        
        # Return default stats if none exist
        return AgentExecutionStats(published_agent_id=published_agent_id)
    
    async def update_execution_stats(self, published_agent_id: str, execution_successful: bool, execution_time: float, error_type: str = None):
        """Update execution statistics"""
        stats = await self.get_execution_stats(published_agent_id)
        
        stats.total_executions += 1
        stats.last_execution_at = datetime.now()
        
        if execution_successful:
            stats.successful_executions += 1
        else:
            stats.failed_executions += 1
            if error_type:
                stats.execution_errors[error_type] = stats.execution_errors.get(error_type, 0) + 1
        
        # Update average execution time
        if stats.average_execution_time:
            stats.average_execution_time = (stats.average_execution_time + execution_time) / 2
        else:
            stats.average_execution_time = execution_time
        
        # Save updated stats
        await self.db.get_collection("agent_execution_stats").replace_one(
            {"_id": str(published_agent_id)},
            stats.model_dump(by_alias=True),
            upsert=True
        )
    
    async def get_published_agent_for_execution(self, published_agent_id: str) -> Optional[PublishedAgent]:
        """Get published agent for execution (with compiled graph)"""
        published_agent = await self.get_published_agent(published_agent_id)
        if not published_agent:
            return None
        
        # Check if compiled graph exists in cache
        if published_agent.compiled_graph_id:
            cached_graph = await get_cache_manager().get_agent_graph(
                published_agent.compiled_graph_id, 
                published_agent.version
            )
            if cached_graph:
                return published_agent
        
        # If no cached graph, recompile
        try:
            draft_agent = await self.agent_service.get_agent(published_agent.original_agent_id)
            if draft_agent:
                compiled_graph = await self.graph_compiler.compile_agent_graph(draft_agent)
                await get_cache_manager().set_agent_graph(str(published_agent_id), published_agent.version, compiled_graph)
                return published_agent
        except Exception as e:
            logger.error(f"Failed to recompile graph for execution: {e}")
        
        return None
