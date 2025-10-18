from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import openai
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, TokenTextSplitter, 
    CharacterTextSplitter
)

import tiktoken
import nltk

from app.database import get_database
from app.models.knowledge_base import (
    KnowledgeBase, KnowledgeBaseStatus, KnowledgeBaseCreateRequest,
    KnowledgeBaseUpdateRequest, Document, DocumentUploadRequest,
    RetrievalRequest, RetrievalResponse, DocumentMetadata,
    ChunkingMethod, EmbeddingModel, ChunkingConfig, RerankConfig
)
from config import settings

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Service for managing knowledge bases and vector storage"""
    
    def __init__(self):
        self.db: AsyncIOMotorDatabase = get_database()
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        openai.api_key = settings.openai_api_key
    
    async def create_knowledge_base(self, request: KnowledgeBaseCreateRequest) -> KnowledgeBase:
        """Create a new knowledge base"""
        kb_id = str(uuid4())
        now = datetime.now()
        
        # Create collection name
        collection_name = f"kb_{kb_id}".replace("-", "_")
        
        # Create Qdrant collection
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI ada-002 embedding size
                distance=Distance.COSINE
            )
        )
        
        # Set default embedding model and chunking config if not provided
        embedding_model = request.embedding_model or EmbeddingModel()
        chunking_config = request.chunking_config or ChunkingConfig()
        
        knowledge_base = KnowledgeBase(
            id=kb_id,
            user_id=request.user_id,
            name=request.name,
            description=request.description,
            status=KnowledgeBaseStatus.ACTIVE,
            collection_name=collection_name,
            embedding_model=embedding_model,
            chunking_config=chunking_config,
            created_at=now,
            updated_at=now
        )
        
        # Save to database
        await self.db.get_collection("knowledge_bases").insert_one(knowledge_base.model_dump(by_alias=True))
        
        logger.info(f"Created knowledge base {kb_id} with collection {collection_name}")
        return knowledge_base
    
    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """Get knowledge base by ID"""
        doc = await self.db.get_collection("knowledge_bases").find_one({"_id": str(kb_id)})
        if doc:
            doc["id"] = doc.pop("_id")
            return KnowledgeBase(**doc)
        return None
    
    async def get_knowledge_bases_by_user(self, user_id: str) -> List[KnowledgeBase]:
        """Get all knowledge bases for a user"""
        cursor = self.db.get_collection("knowledge_bases").find({"user_id": user_id}).sort("created_at", -1)
        knowledge_bases = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            knowledge_bases.append(KnowledgeBase(**doc))
        return knowledge_bases
    
    async def update_knowledge_base(self, kb_id: str, request: KnowledgeBaseUpdateRequest) -> Optional[KnowledgeBase]:
        """Update knowledge base"""
        update_data = {}
        
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.embedding_model is not None:
            update_data["embedding_model"] = request.embedding_model
        if request.chunk_size is not None:
            update_data["chunk_size"] = request.chunk_size
        if request.chunk_overlap is not None:
            update_data["chunk_overlap"] = request.chunk_overlap
        
        update_data["updated_at"] = datetime.now()
        
        result = await self.db.get_collection("knowledge_bases").update_one(
            {"_id": str(kb_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return await self.get_knowledge_base(kb_id)
        
        return None
    
    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete knowledge base and its collection"""
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            return False
        
        # Delete Qdrant collection
        try:
            self.qdrant_client.delete_collection(kb.collection_name)
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant collection {kb.collection_name}: {e}")
        
        # Delete from database
        result = await self.db.get_collection("knowledge_bases").delete_one({"_id": str(kb_id)})
        
        if result.deleted_count > 0:
            logger.info(f"Deleted knowledge base {kb_id}")
            return True
        
        return False
    
    async def upload_document(self, kb_id: str, request: DocumentUploadRequest) -> Document:
        """Upload and process a document"""
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError("Knowledge base not found")
        
        # Create document
        doc_id = str(uuid4())
        document = Document(
            id=doc_id,
            content=request.content,
            metadata=request.metadata
        )
        
        # Split text into chunks based on configuration
        chunks = await self._chunk_text(request.content, kb.chunking_config)
        
        # Generate embeddings for each chunk
        points = []
        for i, chunk in enumerate(chunks):
            # Generate embedding using configured model
            embedding = await self._generate_embedding(chunk, kb.embedding_model)
            
            # Create point for Qdrant
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "document_id": str(doc_id),
                    "chunk_index": i,
                    "content": chunk,
                    "metadata": request.metadata.model_dump()
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=kb.collection_name,
            points=points
        )
        
        # Save document to database
        await self.db.get_collection("knowledge_bases").update_one(
            {"_id": str(kb_id)},
            {"$push": {"documents": document.model_dump()}}
        )
        
        logger.info(f"Uploaded document {doc_id} to knowledge base {kb_id} with {len(chunks)} chunks")
        return document
    
    async def search_knowledge_base(self, kb_id: str, request: RetrievalRequest) -> RetrievalResponse:
        """Search knowledge base"""
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError("Knowledge base not found")
        
        # Generate query embedding using KB's embedding model
        query_embedding = await self._generate_embedding(request.query, kb.embedding_model)
        
        # Determine search parameters
        search_limit = request.top_k
        rerank_config = request.rerank_config
        
        # If reranking is enabled, get more results initially
        if rerank_config and rerank_config.enabled:
            search_limit = max(request.top_k, rerank_config.top_k)
        
        # Search Qdrant
        search_result = self.qdrant_client.search(
            collection_name=kb.collection_name,
            query_vector=query_embedding,
            limit=search_limit,
            score_threshold=request.score_threshold,
            query_filter=models.Filter(**request.filters) if request.filters else None
        )
        
        # Process initial results
        initial_documents = []
        initial_scores = []
        
        for hit in search_result:
            payload = hit.payload
            document = Document(
                id=str(payload["document_id"]),
                content=payload["content"],
                metadata=DocumentMetadata(**payload["metadata"]),
                embedding=hit.vector
            )
            initial_documents.append(document)
            initial_scores.append(hit.score)
        
        # Apply reranking if enabled
        if rerank_config and rerank_config.enabled and initial_documents:
            documents, scores = await self._rerank_documents(
                request.query, initial_documents, initial_scores, rerank_config
            )
        else:
            documents = initial_documents[:request.top_k]
            scores = initial_scores[:request.top_k]
        
        return RetrievalResponse(
            documents=documents,
            scores=scores,
            query_embedding=query_embedding
        )
    
    async def _chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text based on configuration"""
        try:
            if config.method == ChunkingMethod.RECURSIVE_CHARACTER:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    separators=config.separators,
                    length_function=len if config.length_function == "len" else tiktoken.get_encoding("cl100k_base").encode,
                    is_separator_regex=config.is_separator_regex
                )
                return splitter.split_text(text)
            
            elif config.method == ChunkingMethod.TOKEN:
                splitter = TokenTextSplitter(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap
                )
                return splitter.split_text(text)
            
            elif config.method == ChunkingMethod.SENTENCE:
                # Use NLTK for sentence splitting
                try:
                    nltk.download('punkt', quiet=True)
                    sentences = nltk.sent_tokenize(text)
                    chunks = []
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) <= config.chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    return chunks
                except Exception as e:
                    logger.warning(f"NLTK sentence splitting failed, falling back to character splitting: {e}")
                    return await self._chunk_text(text, ChunkingConfig(method=ChunkingMethod.RECURSIVE_CHARACTER, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap))
            
            elif config.method == ChunkingMethod.FIXED_SIZE:
                splitter = CharacterTextSplitter(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    separator=""
                )
                return splitter.split_text(text)
            
            elif config.method == ChunkingMethod.CUSTOM:
                # Custom separators
                if config.custom_separators:
                    separators = config.custom_separators
                else:
                    separators = config.separators
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    separators=separators
                )
                return splitter.split_text(text)
            
            else:
                # Default to recursive character splitting
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap
                )
                return splitter.split_text(text)
                
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            # Fallback to simple splitting
            return [text[i:i+config.chunk_size] for i in range(0, len(text), config.chunk_size - config.chunk_overlap)]
    
    async def _generate_embedding(self, text: str, embedding_model: EmbeddingModel) -> List[float]:
        """Generate embedding for text using configured model"""
        try:
            if embedding_model.provider == "openai":
                response = openai.Embedding.create(
                    input=text,
                    model=embedding_model.model_name,
                    api_key=embedding_model.api_key or settings.openai_api_key
                )
                return response["data"][0]["embedding"]
            
            elif embedding_model.provider == "cohere":
                # Cohere embedding implementation would go here
                raise NotImplementedError("Cohere embeddings not implemented yet")
            
            elif embedding_model.provider == "huggingface":
                # HuggingFace embedding implementation would go here
                raise NotImplementedError("HuggingFace embeddings not implemented yet")
            
            else:
                # Default to OpenAI
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002",
                    api_key=settings.openai_api_key
                )
                return response["data"][0]["embedding"]
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def _rerank_documents(self, query: str, documents: List[Document], scores: List[float], rerank_config: RerankConfig) -> tuple[List[Document], List[float]]:
        """Rerank documents using cross-encoder model"""
        try:
            # For now, implement a simple reranking based on keyword matching
            # In a production system, you would use a proper cross-encoder model
            
            # Take top documents for reranking
            top_documents = documents[:rerank_config.top_k]
            top_scores = scores[:rerank_config.top_k]
            
            # Simple keyword-based reranking
            query_words = set(query.lower().split())
            reranked_docs = []
            reranked_scores = []
            
            for doc, score in zip(top_documents, top_scores):
                doc_words = set(doc.content.lower().split())
                keyword_overlap = len(query_words.intersection(doc_words))
                keyword_ratio = keyword_overlap / len(query_words) if query_words else 0
                
                # Combine semantic score with keyword score
                combined_score = score * 0.7 + keyword_ratio * 0.3
                
                reranked_docs.append(doc)
                reranked_scores.append(combined_score)
            
            # Sort by combined score
            sorted_indices = sorted(range(len(reranked_docs)), key=lambda i: reranked_scores[i], reverse=True)
            
            final_docs = [reranked_docs[i] for i in sorted_indices[:rerank_config.return_top_k]]
            final_scores = [reranked_scores[i] for i in sorted_indices[:rerank_config.return_top_k]]
            
            return final_docs, final_scores
            
        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            # Return original results if reranking fails
            return documents[:rerank_config.return_top_k], scores[:rerank_config.return_top_k]
    
    async def get_collection_stats(self, kb_id: str) -> Dict[str, Any]:
        """Get collection statistics"""
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError("Knowledge base not found")
        
        try:
            info = self.qdrant_client.get_collection(kb.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def batch_upload_documents(self, kb_id: str, requests: List[DocumentUploadRequest]) -> List[Document]:
        """Upload multiple documents in batch"""
        documents = []
        for request in requests:
            document = await self.upload_document(kb_id, request)
            documents.append(document)
        return documents
