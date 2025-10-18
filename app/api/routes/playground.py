from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

from app.services.knowledge_base_service import KnowledgeBaseService
from app.models.knowledge_base import (
    RetrievalRequest, RetrievalResponse, DocumentMetadata,
    ChunkingMethod, EmbeddingModel, ChunkingConfig
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
def get_kb_service() -> KnowledgeBaseService:
    return KnowledgeBaseService()


class PlaygroundTestRequest(BaseModel):
    """Request for playground testing"""
    content: str
    query: str
    chunking_config: Optional[ChunkingConfig] = None
    embedding_model: Optional[EmbeddingModel] = None
    retrieval_config: Optional[Dict[str, Any]] = None


class PlaygroundTestResponse(BaseModel):
    """Response from playground testing"""
    chunks: List[str]
    embeddings_generated: bool
    query_embedding: Optional[List[float]] = None
    retrieval_results: Optional[RetrievalResponse] = None
    test_duration: float
    chunk_count: int
    average_chunk_size: float


@router.post("/chunking/test", response_model=Dict[str, Any])
async def test_chunking(
    content: str,
    chunking_config: ChunkingConfig,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Test different chunking methods on sample content"""
    try:
        import time
        start_time = time.time()
        
        chunks = await kb_service._chunk_text(content, chunking_config)
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        average_size = sum(chunk_sizes) / len(chunks) if chunks else 0
        
        return {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "average_chunk_size": average_size,
            "min_chunk_size": min(chunk_sizes) if chunks else 0,
            "max_chunk_size": max(chunk_sizes) if chunks else 0,
            "chunk_sizes": chunk_sizes,
            "processing_time": time.time() - start_time,
            "config_used": chunking_config.model_dump()
        }
    except Exception as e:
        logger.error(f"Chunking test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking test failed: {str(e)}"
        )


@router.post("/embedding/test", response_model=Dict[str, Any])
async def test_embedding(
    text: str,
    embedding_model: EmbeddingModel,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Test embedding generation with different models"""
    try:
        import time
        start_time = time.time()
        
        embedding = await kb_service._generate_embedding(text, embedding_model)
        
        return {
            "embedding": embedding,
            "embedding_dimensions": len(embedding),
            "text_length": len(text),
            "processing_time": time.time() - start_time,
            "model_used": embedding_model.model_dump()
        }
    except Exception as e:
        logger.error(f"Embedding test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding test failed: {str(e)}"
        )


@router.post("/retrieval/test", response_model=PlaygroundTestResponse)
async def test_retrieval(
    request: PlaygroundTestRequest,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Test complete retrieval pipeline"""
    try:
        import time
        start_time = time.time()
        
        # Use default configs if not provided
        chunking_config = request.chunking_config or ChunkingConfig()
        embedding_model = request.embedding_model or EmbeddingModel()
        
        # Test chunking
        chunks = await kb_service._chunk_text(request.content, chunking_config)
        
        # Test embedding generation
        query_embedding = await kb_service._generate_embedding(request.query, embedding_model)
        
        # Test retrieval (simulated - would need actual KB)
        retrieval_results = None
        embeddings_generated = True
        
        try:
            # This would work if we had a test KB
            # For now, we'll simulate the retrieval
            retrieval_results = RetrievalResponse(
                documents=[],
                scores=[],
                query_embedding=query_embedding
            )
        except Exception as e:
            logger.warning(f"Retrieval simulation failed: {e}")
            embeddings_generated = False
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        average_chunk_size = sum(chunk_sizes) / len(chunks) if chunks else 0
        
        return PlaygroundTestResponse(
            chunks=chunks,
            embeddings_generated=embeddings_generated,
            query_embedding=query_embedding,
            retrieval_results=retrieval_results,
            test_duration=time.time() - start_time,
            chunk_count=len(chunks),
            average_chunk_size=average_chunk_size
        )
        
    except Exception as e:
        logger.error(f"Retrieval test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval test failed: {str(e)}"
        )


@router.post("/kb/{kb_id}/retrieval/test", response_model=Dict[str, Any])
async def test_kb_retrieval(
    kb_id: str,
    query: str,
    retrieval_config: Optional[Dict[str, Any]] = None,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Test retrieval on an actual knowledge base"""
    try:
        import time
        start_time = time.time()
        
        # Create retrieval request
        retrieval_request = RetrievalRequest(
            query=query,
            top_k=retrieval_config.get("top_k", 5) if retrieval_config else 5,
            score_threshold=retrieval_config.get("score_threshold", 0.7) if retrieval_config else 0.7,
            filters=retrieval_config.get("filters") if retrieval_config else None
        )
        
        # Perform retrieval
        results = await kb_service.search_knowledge_base(kb_id, retrieval_request)
        
        return {
            "query": query,
            "results_count": len(results.documents),
            "documents": [
                {
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "score": score,
                    "metadata": doc.metadata.model_dump()
                }
                for doc, score in zip(results.documents, results.scores)
            ],
            "scores": results.scores,
            "query_embedding_dimensions": len(results.query_embedding),
            "retrieval_time": time.time() - start_time,
            "config_used": retrieval_config
        }
        
    except Exception as e:
        logger.error(f"KB retrieval test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KB retrieval test failed: {str(e)}"
        )


@router.get("/chunking/methods", response_model=List[str])
async def get_chunking_methods():
    """Get available chunking methods"""
    return [method.value for method in ChunkingMethod]


@router.get("/embedding/models", response_model=List[Dict[str, Any]])
async def get_embedding_models():
    """Get available embedding models"""
    return [
        {
            "provider": "openai",
            "model_name": "text-embedding-ada-002",
            "dimensions": 1536,
            "max_tokens": 8191
        },
        {
            "provider": "openai", 
            "model_name": "text-embedding-3-small",
            "dimensions": 1536,
            "max_tokens": 8191
        },
        {
            "provider": "openai",
            "model_name": "text-embedding-3-large", 
            "dimensions": 3072,
            "max_tokens": 8191
        }
    ]


@router.post("/compare/configurations", response_model=Dict[str, Any])
async def compare_configurations(
    content: str,
    query: str,
    configurations: List[Dict[str, Any]],  # List of chunking/embedding configs to compare
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Compare different chunking and embedding configurations"""
    try:
        import time
        
        results = []
        
        for i, config in enumerate(configurations):
            start_time = time.time()
            
            chunking_config = ChunkingConfig(**config.get("chunking", {}))
            embedding_model = EmbeddingModel(**config.get("embedding", {}))
            
            # Test chunking
            chunks = await kb_service._chunk_text(content, chunking_config)
            
            # Test embedding
            query_embedding = await kb_service._generate_embedding(query, embedding_model)
            
            chunk_sizes = [len(chunk) for chunk in chunks]
            
            results.append({
                "config_index": i,
                "config": config,
                "chunk_count": len(chunks),
                "average_chunk_size": sum(chunk_sizes) / len(chunks) if chunks else 0,
                "min_chunk_size": min(chunk_sizes) if chunks else 0,
                "max_chunk_size": max(chunk_sizes) if chunks else 0,
                "embedding_dimensions": len(query_embedding),
                "processing_time": time.time() - start_time,
                "sample_chunks": chunks[:3]  # First 3 chunks as samples
            })
        
        return {
            "comparison_results": results,
            "best_chunk_count": max(results, key=lambda x: x["chunk_count"]),
            "fastest_processing": min(results, key=lambda x: x["processing_time"]),
            "most_efficient": min(results, key=lambda x: x["processing_time"] / x["chunk_count"])
        }
        
    except Exception as e:
        logger.error(f"Configuration comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration comparison failed: {str(e)}"
        )
