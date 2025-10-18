from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class KnowledgeBaseStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROCESSING = "processing"
    ERROR = "error"


class RerankConfig(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 10  # Number of documents to rerank
    return_top_k: int = 5  # Final number of documents to return


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.7
    search_type: str = "similarity"  # similarity, mmr, etc.
    rerank_config: Optional[RerankConfig] = None
    hybrid_search: bool = False  # Combine keyword and semantic search
    alpha: float = 0.5  # Weight for hybrid search (0 = pure semantic, 1 = pure keyword)


class DocumentMetadata(BaseModel):
    source: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: List[str] = []


class Document(BaseModel):
    id: str = Field(..., alias="_id")
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class KnowledgeBaseItem(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    enabled: bool = True
    priority: int = 1
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    class Config:
        use_enum_values = True
        populate_by_name = True


class ChunkingMethod(str, Enum):
    RECURSIVE_CHARACTER = "recursive_character"
    TOKEN = "token"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    CUSTOM = "custom"


class EmbeddingModel(BaseModel):
    provider: str = "openai"
    model_name: str = "text-embedding-ada-002"
    dimensions: Optional[int] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None


class ChunkingConfig(BaseModel):
    method: ChunkingMethod = ChunkingMethod.RECURSIVE_CHARACTER
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ["\n\n", "\n", " ", ""]
    length_function: str = "len"
    is_separator_regex: bool = False
    custom_separators: Optional[List[str]] = None


class KnowledgeBase(BaseModel):
    id: str = Field(..., alias="_id")
    user_id: str
    name: str
    description: str
    status: KnowledgeBaseStatus = KnowledgeBaseStatus.ACTIVE
    collection_name: str
    documents: List[Document] = []
    embedding_model: EmbeddingModel = Field(default_factory=lambda: EmbeddingModel())
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    created_at: datetime
    updated_at: datetime

    class Config:
        use_enum_values = True
        populate_by_name = True


class KnowledgeBaseCreateRequest(BaseModel):
    name: str
    description: str
    user_id: str
    embedding_model: Optional[EmbeddingModel] = None
    chunking_config: Optional[ChunkingConfig] = None


class KnowledgeBaseUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    embedding_model: Optional[EmbeddingModel] = None
    chunking_config: Optional[ChunkingConfig] = None


class DocumentUploadRequest(BaseModel):
    content: str
    metadata: DocumentMetadata


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None
    rerank_config: Optional[RerankConfig] = None
    hybrid_search: bool = False
    alpha: float = 0.5


class RetrievalResponse(BaseModel):
    documents: List[Document]
    scores: List[float]
    query_embedding: List[float]
