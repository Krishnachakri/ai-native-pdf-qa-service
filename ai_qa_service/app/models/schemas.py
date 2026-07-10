from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Machine-readable error code", examples=["EMPTY_DOCUMENT"])
    message: str = Field(..., description="Human-readable error explanation", examples=["Uploaded PDF contains no extractable text."])

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status", examples=["healthy"])
    vector_db: str = Field(..., description="ChromaDB connection status", examples=["connected"])
    embedding_model: str = Field(..., description="Embedding model status", examples=["loaded"])
    llm: str = Field(..., description="LLM provider configuration status", examples=["configured"])

class Citation(BaseModel):
    page_number: int = Field(..., description="1-based index of the page from which the citation was extracted", examples=[1])
    excerpt: str = Field(..., description="Exact snippet of text retrieved from the document", examples=["FastAPI is a modern, fast (high-performance), web framework for building APIs."])

class StructuredQAResponse(BaseModel):
    answer: str = Field(..., description="The generated answer to the user's question, sourced solely from the context.", examples=["FastAPI is a web framework for building APIs with Python."])
    citations: List[Citation] = Field(default_factory=list, description="A list of exact quotes with their respective page numbers validating the answer.")
    found: bool = Field(..., description="Whether the answer was successfully resolved from the retrieved document context.", examples=[True])
    confidence: ConfidenceLevel = Field(..., description="Confidence level based on retrieved context relevance score.", examples=[ConfidenceLevel.HIGH])

class ProcessingMetrics(BaseModel):
    parse_time_ms: int = Field(..., description="Time taken to extract text from PDF in milliseconds", examples=[120])
    chunking_time_ms: int = Field(..., description="Time taken to token-chunk the text in milliseconds", examples=[15])
    embedding_time_ms: int = Field(..., description="Time taken to generate embeddings in milliseconds", examples=[850])
    indexing_time_ms: int = Field(..., description="Time taken to index chunks in ChromaDB in milliseconds", examples=[110])
    total_time_ms: int = Field(..., description="Total processing time in milliseconds", examples=[1095])

class UploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique document ID (hash-based)", examples=["doc_8f3d1b6a7c8e"])
    filename: str = Field(..., description="Original name of the uploaded file", examples=["fastapi_tutorial.pdf"])
    upload_time: str = Field(..., description="ISO 8601 timestamp of upload", examples=["2026-07-10T12:00:00Z"])
    hash: str = Field(..., description="SHA-256 hash of the document content", examples=["a8fd0c9d8e7b6a5c4d3e2f1b0a9f8e7d"])
    file_size: int = Field(..., description="File size in bytes", examples=[102400])
    pages: int = Field(..., description="Total number of pages extracted", examples=[12])
    chunks: int = Field(..., description="Number of vector chunks indexed", examples=[35])
    embedding_model: str = Field(..., description="Embedding model name used", examples=["all-MiniLM-L6-v2"])
    processing_time_ms: int = Field(..., description="Total processing latency in milliseconds", examples=[1095])
    metrics: ProcessingMetrics = Field(..., description="Granular processing performance metrics")

class DocumentResponse(BaseModel):
    document_id: str = Field(..., description="Unique document ID (hash-based)", examples=["doc_8f3d1b6a7c8e"])
    filename: str = Field(..., description="Original name of the uploaded file", examples=["fastapi_tutorial.pdf"])
    upload_time: str = Field(..., description="ISO 8601 timestamp of upload", examples=["2026-07-10T12:00:00Z"])
    hash: str = Field(..., description="SHA-256 hash of the document content", examples=["a8fd0c9d8e7b6a5c4d3e2f1b0a9f8e7d"])
    file_size: int = Field(..., description="File size in bytes", examples=[102400])
    pages: int = Field(..., description="Total number of pages extracted", examples=[12])
    chunks: int = Field(..., description="Number of vector chunks indexed", examples=[35])
    embedding_model: str = Field(..., description="Embedding model name used", examples=["all-MiniLM-L6-v2"])

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the document.", examples=["What is the maximum file size?"])
