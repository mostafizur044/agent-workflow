from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
import logging

from app.services.knowledge_base_service import KnowledgeBaseService
from app.models.knowledge_base import (
    KnowledgeBase, KnowledgeBaseCreateRequest, KnowledgeBaseUpdateRequest,
    DocumentUploadRequest, RetrievalRequest, RetrievalResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
def get_kb_service() -> KnowledgeBaseService:
    return KnowledgeBaseService()


@router.post("/", response_model=KnowledgeBase, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Create a new knowledge base"""
    try:
        kb = await kb_service.create_knowledge_base(request)
        return kb
    except Exception as e:
        logger.error(f"Failed to create knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create knowledge base"
        )


@router.get("/", response_model=List[KnowledgeBase])
async def get_knowledge_bases(
    user_id: str,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Get all knowledge bases for a user"""
    try:
        kbs = await kb_service.get_knowledge_bases_by_user(user_id)
        return kbs
    except Exception as e:
        logger.error(f"Failed to get knowledge bases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve knowledge bases"
        )


@router.get("/{kb_id}", response_model=KnowledgeBase)
async def get_knowledge_base(
    kb_id: str,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Get knowledge base by ID"""
    kb = await kb_service.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
    return kb


@router.put("/{kb_id}", response_model=KnowledgeBase)
async def update_knowledge_base(
    kb_id: str,
    request: KnowledgeBaseUpdateRequest,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Update knowledge base"""
    kb = await kb_service.update_knowledge_base(kb_id, request)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )
    return kb


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb_id: str,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Delete knowledge base"""
    success = await kb_service.delete_knowledge_base(kb_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found"
        )


@router.post("/{kb_id}/documents", response_model=dict)
async def upload_document(
    kb_id: str,
    request: DocumentUploadRequest,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Upload document to knowledge base"""
    try:
        document = await kb_service.upload_document(kb_id, request)
        return {"document_id": str(document.id), "message": "Document uploaded successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )


@router.post("/{kb_id}/search", response_model=RetrievalResponse)
async def search_knowledge_base(
    kb_id: str,
    request: RetrievalRequest,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Search knowledge base"""
    try:
        results = await kb_service.search_knowledge_base(kb_id, request)
        return results
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to search knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search knowledge base"
        )


@router.get("/{kb_id}/stats", response_model=dict)
async def get_knowledge_base_stats(
    kb_id: str,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Get knowledge base statistics"""
    try:
        stats = await kb_service.get_collection_stats(kb_id)
        return stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get knowledge base stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get knowledge base stats"
        )
