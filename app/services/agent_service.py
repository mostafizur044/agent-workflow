from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

from app.core.cache import get_cache_manager
from app.database import get_database
from app.models.agent import (
    Agent, AgentConfig, AgentStatus, AgentCreateRequest, 
    AgentUpdateRequest, AgentDependency, 
    AgentVersion
)


logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing agent configurations and operations"""
    
    def __init__(self):
        self.db: AsyncIOMotorDatabase = get_database()
    
    async def create_agent(self, request: AgentCreateRequest) -> Agent:
        """Create a new agent with default configuration"""
        agent_id = str(uuid4())
        now = datetime.now()
        
        # Create default configuration
        default_config = AgentConfig(
            metadata={
                "created_at": now,
                "updated_at": now,
                "created_by": request.user_id
            }
        )
        
        agent = Agent(
            id=agent_id,
            user_id=request.user_id,
            name=request.name,
            description=request.description,
            version=1,
            status=AgentStatus.DRAFT,
            config=default_config,
            created_at=now,
            updated_at=now
        )
        
        # Save to database
        await self.db.get_collection("agents").insert_one(agent.model_dump(by_alias=True))
        
        # Create initial version
        await self.create_agent_version(agent_id, 1, default_config)
        
        logger.info(f"Created agent {agent_id} for user {request.user_id}")
        return agent
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        doc = await self.db.get_collection("agents").find_one({"_id": str(agent_id)})
        if doc:
            doc["id"] = doc.pop("_id")
            return Agent(**doc)
        return None
    
    async def get_agents_by_user(self, user_id: str) -> List[Agent]:
        """Get all agents for a user"""
        cursor = self.db.get_collection("agents").find({"user_id": user_id}).sort("created_at", -1)
        agents = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            agents.append(Agent(**doc))
        return agents
    
    async def update_agent(self, agent_id: str, request: AgentUpdateRequest) -> Optional[Agent]:
        """Update agent configuration"""
        update_data = {}
        
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.config is not None:
            cfg_dict = request.config.model_dump()
            cfg_dict["metadata"]["updated_at"] = datetime.now()
            update_data["config"] = cfg_dict
        
        update_data["updated_at"] = datetime.now()
        
        result = await self.db.get_collection("agents").update_one(
            {"_id": str(agent_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            # Invalidate cache
            await get_cache_manager().invalidate_agent_cache(str(agent_id))
            return await self.get_agent(agent_id)
        
        return None

    
    
    async def publish_agent(self, agent_id: str) -> Optional[Agent]:
        """Publish agent (mark as published)"""
        now = datetime.now()
        
        result = await self.db.get_collection("agents").update_one(
            {"_id": str(agent_id)},
            {
                "$set": {
                    "status": AgentStatus.PUBLISHED,
                    "published_at": now,
                    "updated_at": now,
                    "config.metadata.published_at": now
                }
            }
        )
        
        if result.modified_count > 0:
            # Invalidate cache to force recompilation
            await get_cache_manager().invalidate_agent_cache(str(agent_id))
            return await self.get_agent(agent_id)
        
        return None
    
    async def archive_agent(self, agent_id: str) -> Optional[Agent]:
        """Archive agent"""
        result = await self.db.get_collection("agents").update_one(
            {"_id": str(agent_id)},
            {
                "$set": {
                    "status": AgentStatus.ARCHIVED,
                    "updated_at": datetime.now()
                }
            }
        )
        
        if result.modified_count > 0:
            await get_cache_manager().invalidate_agent_cache(str(agent_id))
            return await self.get_agent(agent_id)
        
        return None
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete agent and all related data"""
        # Delete agent
        result = await self.db.get_collection("agents").delete_one({"_id": str(agent_id)})
        
        if result.deleted_count > 0:
            # Delete related data
            await self.db.get_collection("agent_dependencies").delete_many({"agent_id": str(agent_id)})
            await self.db.get_collection("agent_versions").delete_many({"agent_id": str(agent_id)})
            await self.db.get_collection("agent_links").delete_many({
                "$or": [
                    {"parent_agent_id": str(agent_id)},
                    {"child_agent_id": str(agent_id)}
                ]
            })
            
            # Invalidate cache
            await get_cache_manager().invalidate_agent_cache(str(agent_id))
            
            logger.info(f"Deleted agent {agent_id} and related data")
            return True
        
        return False
    
    async def add_knowledge_base(self, agent_id: str, kb_id: str, priority: int = 1) -> Optional[Agent]:
        """Add knowledge base to agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        # Add KB to configuration
        kb_item = {
            "id": str(kb_id),
            "name": f"KB_{kb_id}",  # You might want to fetch the actual name
            "enabled": True,
            "priority": priority,
            "retrieval_config": {
                "top_k": 5,
                "score_threshold": 0.7,
                "search_type": "similarity"
            }
        }
        
        config_dict = agent.config.model_dump()
        config_dict["knowledge_bases"]["items"].append(kb_item)
        config_dict["knowledge_bases"]["enabled"] = True
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def add_tool(self, agent_id: str, tool_id: str) -> Optional[Agent]:
        """Add tool to agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        # Add tool to configuration
        tool_item = {
            "id": str(tool_id),
            "name": f"Tool_{tool_id}",  # You might want to fetch the actual name
            "type": "api",
            "enabled": True,
            "config": {}
        }
        
        config_dict = agent.config.model_dump()
        config_dict["tools"]["items"].append(tool_item)
        config_dict["tools"]["enabled"] = True
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def add_sub_agent(self, agent_id: str, sub_agent_id: str, trigger_condition: str = None) -> Optional[Agent]:
        """Add sub-agent to agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        # Add sub-agent to configuration
        sub_agent_item = {
            "id": str(sub_agent_id),
            "name": f"SubAgent_{sub_agent_id}",  # You might want to fetch the actual name
            "enabled": True,
            "trigger_condition": trigger_condition
        }
        
        config_dict = agent.config.model_dump()
        config_dict["sub_agents"]["items"].append(sub_agent_item)
        config_dict["sub_agents"]["enabled"] = True
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def create_agent_version(self, agent_id: str, version: int, config: AgentConfig) -> AgentVersion:
        """Create a new agent version snapshot"""
        version_id = str(uuid4())
        now = datetime.now()
        
        agent_version = AgentVersion(
            id=version_id,
            agent_id=agent_id,
            version=version,
            config_snapshot=config,
            created_at=now
        )
        
        await self.db.get_collection("agent_versions").insert_one(agent_version.model_dump(by_alias=True))
        return agent_version
    
    async def get_agent_versions(self, agent_id: str) -> List[AgentVersion]:
        """Get all versions for an agent"""
        cursor = self.db.get_collection("agent_versions").find({"agent_id": str(agent_id)}).sort("version", -1)
        versions = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            versions.append(AgentVersion(**doc))
        return versions
    
    async def get_agent_dependencies(self, agent_id: str) -> List[AgentDependency]:
        """Get all dependencies for an agent"""
        cursor = self.db.get_collection("agent_dependencies").find({"agent_id": str(agent_id)})
        dependencies = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            dependencies.append(AgentDependency(**doc))
        return dependencies
    
    async def validate_agent_dependencies(self, agent_id: str) -> Dict[str, List[str]]:
        """Validate that all enabled dependencies exist and are accessible"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return {"error": ["Agent not found"]}
        
        errors = []
        warnings = []
        
        # Validate knowledge bases
        if agent.config.knowledge_bases.enabled:
            for kb_item in agent.config.knowledge_bases.items:
                if kb_item.enabled:
                    # Check if KB exists and is active
                    kb = await self.db.get_collection("knowledge_bases").find_one({
                        "_id": str(kb_item.id),
                        "status": "active"
                    })
                    if not kb:
                        errors.append(f"Knowledge base {kb_item.id} not found or inactive")
        
        # Validate tools
        if agent.config.tools.enabled:
            for tool_item in agent.config.tools.items:
                if tool_item.enabled:
                    # Check if tool exists and is active
                    tool = await self.db.get_collection("tools").find_one({
                        "_id": str(tool_item.id),
                        "status": "active"
                    })
                    if not tool:
                        errors.append(f"Tool {tool_item.id} not found or inactive")
        
        # Validate sub-agents
        if agent.config.sub_agents.enabled:
            for sub_agent_item in agent.config.sub_agents.items:
                if sub_agent_item.enabled:
                    # Check if sub-agent exists and is published
                    sub_agent = await self.db.get_collection("agents").find_one({
                        "_id": str(sub_agent_item.id),
                        "status": "published"
                    })
                    if not sub_agent:
                        errors.append(f"Sub-agent {sub_agent_item.id} not found or not published")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0
        }
    
    async def remove_knowledge_base(self, agent_id: str, kb_id: str) -> Optional[Agent]:
        """Remove knowledge base from agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        # Remove KB from items list
        config_dict["knowledge_bases"]["items"] = [
            item for item in config_dict["knowledge_bases"]["items"] 
            if item["id"] != str(kb_id)
        ]
        
        # If no KBs left, disable knowledge bases
        if not config_dict["knowledge_bases"]["items"]:
            config_dict["knowledge_bases"]["enabled"] = False
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def remove_tool(self, agent_id: str, tool_id: str) -> Optional[Agent]:
        """Remove tool from agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        # Remove tool from items list
        config_dict["tools"]["items"] = [
            item for item in config_dict["tools"]["items"] 
            if item["id"] != str(tool_id)
        ]
        
        # If no tools left, disable tools
        if not config_dict["tools"]["items"]:
            config_dict["tools"]["enabled"] = False
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def remove_sub_agent(self, agent_id: str, sub_agent_id: str) -> Optional[Agent]:
        """Remove sub-agent from agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        # Remove sub-agent from items list
        config_dict["sub_agents"]["items"] = [
            item for item in config_dict["sub_agents"]["items"] 
            if item["id"] != str(sub_agent_id)
        ]
        
        # If no sub-agents left, disable sub-agents
        if not config_dict["sub_agents"]["items"]:
            config_dict["sub_agents"]["enabled"] = False
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def toggle_knowledge_base(self, agent_id: str, kb_id: str, enabled: bool) -> Optional[Agent]:
        """Toggle specific knowledge base in agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        
        # Find and update the specific KB
        for item in config_dict["knowledge_bases"]["items"]:
            if item["id"] == str(kb_id):
                item["enabled"] = enabled
                break
        
        # If enabling a KB, also enable the knowledge_bases section
        if enabled:
            config_dict["knowledge_bases"]["enabled"] = True
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def toggle_tool(self, agent_id: str, tool_id: str, enabled: bool) -> Optional[Agent]:
        """Toggle specific tool in agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        
        # Find and update the specific tool
        for item in config_dict["tools"]["items"]:
            if item["id"] == str(tool_id):
                item["enabled"] = enabled
                break
        
        # If enabling a tool, also enable the tools section
        if enabled:
            config_dict["tools"]["enabled"] = True
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def toggle_sub_agent(self, agent_id: str, sub_agent_id: str, enabled: bool) -> Optional[Agent]:
        """Toggle specific sub-agent in agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        
        # Find and update the specific sub-agent
        for item in config_dict["sub_agents"]["items"]:
            if item["id"] == str(sub_agent_id):
                item["enabled"] = enabled
                break
        
        # If enabling a sub-agent, also enable the sub_agents section
        if enabled:
            config_dict["sub_agents"]["enabled"] = True
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def update_knowledge_base_config(self, agent_id: str, kb_id: str, config: Dict[str, Any]) -> Optional[Agent]:
        """Update knowledge base configuration in agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        
        # Find and update the specific KB config
        for item in config_dict["knowledge_bases"]["items"]:
            if item["id"] == str(kb_id):
                item["retrieval_config"] = config
                break
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def update_tool_config(self, agent_id: str, tool_id: str, config: Dict[str, Any]) -> Optional[Agent]:
        """Update tool configuration in agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        
        # Find and update the specific tool config
        for item in config_dict["tools"]["items"]:
            if item["id"] == str(tool_id):
                item["config"] = config
                break
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
    
    async def update_sub_agent_config(self, agent_id: str, sub_agent_id: str, trigger_condition: str) -> Optional[Agent]:
        """Update sub-agent configuration in agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None
        
        config_dict = agent.config.model_dump()
        
        # Find and update the specific sub-agent config
        for item in config_dict["sub_agents"]["items"]:
            if item["id"] == str(sub_agent_id):
                item["trigger_condition"] = trigger_condition
                break
        
        update_request = AgentUpdateRequest(config=AgentConfig(**config_dict))
        return await self.update_agent(agent_id, update_request)
