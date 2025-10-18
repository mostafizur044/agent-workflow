from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.database import init_database
from app.api.routes import agents, knowledge_bases, tools, execution, models, agent_publish, playground
from app.core.cache import CacheManager, init_cache
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    await init_cache()
    
    yield
    # Shutdown
    pass


app = FastAPI(
    title="AI Agent Workflow System",
    description="Dynamic AI agent configuration and execution platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(knowledge_bases.router, prefix="/api/knowledge-bases", tags=["knowledge-bases"])
app.include_router(tools.router, prefix="/api/tools", tags=["tools"])
app.include_router(execution.router, prefix="/api/execution", tags=["execution"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(agent_publish.router, prefix="/api/publish", tags=["agent-publish"])
app.include_router(playground.router, prefix="/api/playground", tags=["playground"])


@app.get("/")
async def root():
    return {"message": "AI Agent Workflow System API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
