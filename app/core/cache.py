import redis.asyncio as redis
import json
import pickle
import logging
from typing import Optional, Any, Dict, List

from config import settings

logger = logging.getLogger(__name__)

# Global Redis connection
redis_client: Optional[redis.Redis] = None

async def init_cache():
    """Initialize Redis cache connection"""
    global redis_client
    
    try:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True, password=settings.redis_pass, username="default")
        await redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise


def get_cache() -> redis.Redis:
    """Get Redis cache instance"""
    if not redis_client:
        raise RuntimeError("Redis cache not initialized")
    return redis_client


class CacheManager:
    """Redis cache manager for agent graphs and metadata"""
    
    def __init__(self):
        self.redis = get_cache()
    
    async def set_agent_graph(self, agent_id: str, version: int, graph_data: Any, ttl: int = 3600):
        """Cache compiled agent graph"""
        key = f"agent:{agent_id}:v{version}:graph"
        serialized = pickle.dumps(graph_data)
        await self.redis.setex(key, ttl, serialized)
        logger.info(f"Cached graph for agent {agent_id} version {version}")
    
    async def get_agent_graph(self, agent_id: str, version: int) -> Optional[Any]:
        """Retrieve cached agent graph"""
        key = f"agent:{agent_id}:v{version}:graph"
        data = await self.redis.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    async def set_agent_metadata(self, agent_id: str, version: int, metadata: Dict[str, Any], ttl: int = 3600):
        """Cache agent metadata"""
        key = f"agent:{agent_id}:v{version}:metadata"
        await self.redis.setex(key, ttl, json.dumps(metadata, default=str))
    
    async def get_agent_metadata(self, agent_id: str, version: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached agent metadata"""
        key = f"agent:{agent_id}:v{version}:metadata"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def set_subgraphs(self, agent_id: str, version: int, subgraphs: List[Dict[str, Any]], ttl: int = 3600):
        """Cache sub-agent graphs"""
        key = f"agent:{agent_id}:v{version}:subgraphs"
        await self.redis.setex(key, ttl, json.dumps(subgraphs, default=str))
    
    async def get_subgraphs(self, agent_id: str, version: int) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached sub-agent graphs"""
        key = f"agent:{agent_id}:v{version}:subgraphs"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def set_dependencies(self, agent_id: str, dependencies: Dict[str, List[str]], ttl: int = 3600):
        """Cache agent dependencies"""
        key = f"agent:{agent_id}:dependencies"
        await self.redis.setex(key, ttl, json.dumps(dependencies))
    
    async def get_dependencies(self, agent_id: str) -> Optional[Dict[str, List[str]]]:
        """Retrieve cached agent dependencies"""
        key = f"agent:{agent_id}:dependencies"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def invalidate_agent_cache(self, agent_id: str):
        """Invalidate all cache entries for an agent"""
        pattern = f"agent:{agent_id}:*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries for agent {agent_id}")
    
    async def set_execution_trace(self, execution_id: str, trace: Dict[str, Any], ttl: int = 86400):
        """Cache execution trace for debugging"""
        key = f"execution:{execution_id}:trace"
        await self.redis.setex(key, ttl, json.dumps(trace, default=str))
    
    async def get_execution_trace(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached execution trace"""
        key = f"execution:{execution_id}:trace"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def set_session_state(self, session_id: str, state: Dict[str, Any], ttl: int = 3600):
        """Cache session state for conversation continuity"""
        key = f"session:{session_id}:state"
        await self.redis.setex(key, ttl, json.dumps(state, default=str))
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached session state"""
        key = f"session:{session_id}:state"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def close(self):
        """Close Redis connection"""
        await self.redis.close()


cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    """Get CacheManager instance"""
    global cache_manager
    if not cache_manager:
        cache_manager = CacheManager()
    return cache_manager