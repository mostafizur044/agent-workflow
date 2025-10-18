from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, TEXT
from typing import Optional
import logging

from config import settings

logger = logging.getLogger(__name__)

# Global database connection
client: Optional[AsyncIOMotorClient] = None
database: Optional[AsyncIOMotorDatabase] = None


async def init_database():
    """Initialize database connection and create indexes"""
    global client, database
    
    try:
        client = AsyncIOMotorClient(settings.mongodb_url)
        database = client[settings.mongodb_database]
        
        # Test connection
        await client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def create_indexes():
    """Create database indexes for optimal performance"""
    if database is None:
        return
    
    # Agents collection indexes
    agents_indexes = [
        IndexModel([("user_id", ASCENDING)]),
        IndexModel([("status", ASCENDING)]),
        IndexModel([("name", ASCENDING)]),
        IndexModel([("created_at", ASCENDING)]),
        IndexModel([("updated_at", ASCENDING)]),
    ]
    
    await database.agents.create_indexes(agents_indexes)
    
    # Agent dependencies indexes
    dependencies_indexes = [
        IndexModel([("agent_id", ASCENDING)]),
        IndexModel([("dependency_type", ASCENDING)]),
        IndexModel([("dependency_id", ASCENDING)]),
        IndexModel([("agent_id", ASCENDING), ("dependency_type", ASCENDING)]),
    ]
    
    await database.agent_dependencies.create_indexes(dependencies_indexes)
    
    # Agent versions indexes
    versions_indexes = [
        IndexModel([("agent_id", ASCENDING)]),
        IndexModel([("agent_id", ASCENDING), ("version", ASCENDING)]),
    ]
    
    await database.agent_versions.create_indexes(versions_indexes)
    
    # Agent links indexes (for sub-agents)
    links_indexes = [
        IndexModel([("parent_agent_id", ASCENDING)]),
        IndexModel([("child_agent_id", ASCENDING)]),
        IndexModel([("parent_agent_id", ASCENDING), ("enabled", ASCENDING)]),
    ]
    
    await database.agent_links.create_indexes(links_indexes)
    
    # Knowledge bases indexes
    kb_indexes = [
        IndexModel([("user_id", ASCENDING)]),
        IndexModel([("name", TEXT)]),
        IndexModel([("status", ASCENDING)]),
    ]
    
    await database.knowledge_bases.create_indexes(kb_indexes)
    
    # Tools indexes
    tools_indexes = [
        IndexModel([("user_id", ASCENDING)]),
        IndexModel([("name", TEXT)]),
        IndexModel([("type", ASCENDING)]),
        IndexModel([("status", ASCENDING)]),
    ]
    
    await database.tools.create_indexes(tools_indexes)
    
    # Models indexes
    models_indexes = [
        IndexModel([("user_id", ASCENDING)]),
        IndexModel([("name", TEXT)]),
        IndexModel([("provider", ASCENDING)]),
        IndexModel([("model_type", ASCENDING)]),
        IndexModel([("is_active", ASCENDING)]),
        IndexModel([("is_shared", ASCENDING)]),
    ]
    
    await database.models.create_indexes(models_indexes)
    
    # Published agents indexes
    published_agents_indexes = [
        IndexModel([("user_id", ASCENDING)]),
        IndexModel([("original_agent_id", ASCENDING)]),
        IndexModel([("status", ASCENDING)]),
        IndexModel([("published_at", ASCENDING)]),
        IndexModel([("name", TEXT)]),
    ]
    
    await database.published_agents.create_indexes(published_agents_indexes)
    
    # Agent publish history indexes
    publish_history_indexes = [
        IndexModel([("agent_id", ASCENDING)]),
        IndexModel([("published_agent_id", ASCENDING)]),
        IndexModel([("published_at", ASCENDING)]),
        IndexModel([("version", ASCENDING)]),
    ]
    
    await database.agent_publish_history.create_indexes(publish_history_indexes)
    
    # Agent execution stats indexes
    execution_stats_indexes = [
        IndexModel([("published_agent_id", ASCENDING)]),
        IndexModel([("total_executions", ASCENDING)]),
        IndexModel([("last_execution_at", ASCENDING)]),
    ]
    
    await database.agent_execution_stats.create_indexes(execution_stats_indexes)
    
    logger.info("Database indexes created successfully")


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    if database is None:
        raise RuntimeError("Database not initialized")
    return database


async def close_database():
    """Close database connection"""
    global client
    if client:
        client.close()
        logger.info("Database connection closed")
