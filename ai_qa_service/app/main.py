import time
import hashlib
import uuid
from datetime import datetime, timezone
from typing import List
from fastapi import FastAPI, UploadFile, File, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.logging import setup_logging, logger, request_id_ctx, document_id_ctx
from app.core.middleware import RequestIDMiddleware
from app.core.security import validate_pdf_file, FileTooLargeError, InvalidPDFSignatureError
from app.services.pdf_parser import parse_pdf, EmptyDocumentError, OCRNotSupportedError
from app.services.chunker import PageAwareChunker
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.models.schemas import (
    UploadResponse,
    DocumentResponse,
    QueryRequest,
    StructuredQAResponse,
    HealthResponse,
    ProcessingMetrics,
    ErrorResponse,
    ConfidenceLevel
)

# 1. Initialize FastAPI Application
app = FastAPI(
    title="RAG PDF Q&A Service",
    version="1.0.0",
    description="Production-grade Retrieval-Augmented Generation Q&A backend for PDF documents.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 2. Setup Logging Configuration
setup_logging()

# 3. Register Middleware
app.add_middleware(RequestIDMiddleware)

# 4. Instantiate Services (Warmup)
vector_store = VectorStore()
embedder = Embedder()
llm_service = LLMService()

# 5. Global Exception Handlers for Structured JSON Error Responses
@app.exception_handler(FileTooLargeError)
async def file_too_large_handler(request: Request, exc: FileTooLargeError):
    logger.error("Upload failed: File too large", extra={"extra_fields": {"size": exc.size, "max_size": exc.max_size}})
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content={
            "error": "FILE_TOO_LARGE",
            "message": f"Uploaded file exceeds the maximum allowed size of {exc.max_size / (1024 * 1024):.1f}MB."
        }
    )

@app.exception_handler(InvalidPDFSignatureError)
async def invalid_pdf_signature_handler(request: Request, exc: InvalidPDFSignatureError):
    logger.error("Upload failed: Invalid PDF signature", extra={"extra_fields": {"signature": str(exc.signature)}})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "INVALID_PDF",
            "message": "Uploaded file is not a valid PDF document (magic bytes signature mismatch)."
        }
    )

@app.exception_handler(EmptyDocumentError)
async def empty_document_handler(request: Request, exc: EmptyDocumentError):
    logger.error("Upload failed: Empty document")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "EMPTY_DOCUMENT",
            "message": str(exc)
        }
    )

@app.exception_handler(OCRNotSupportedError)
async def ocr_not_supported_handler(request: Request, exc: OCRNotSupportedError):
    logger.error("Upload failed: OCR not supported")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "OCR_NOT_SUPPORTED",
            "message": str(exc)
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"Value error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "INVALID_PDF",
            "message": str(exc)
        }
    )

class DocumentNotFoundError(Exception):
    """Custom exception raised when a requested document ID does not exist."""
    pass

@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "DOCUMENT_NOT_FOUND",
            "message": "Document with the specified ID was not found."
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = f"Validation failed: {errors[0].get('loc')}: {errors[0].get('msg')}" if errors else "Validation error."
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": msg
        }
    )

# 6. API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """Checks the health and availability of dependencies (Vector Store, Embedder, LLM Provider)."""
    try:
        # Check ChromaDB connection status
        _ = vector_store.collection.count()
        vector_db_status = "connected"
    except Exception as e:
        logger.error(f"Health Check: ChromaDB check failed: {e}")
        vector_db_status = "disconnected"
        
    embedding_status = "loaded" if (embedder.model is not None) else "unloaded"
    llm_status = "configured" if settings.OPENAI_API_KEY else "misconfigured"
    
    overall_status = "healthy" if (
        vector_db_status == "connected" and 
        embedding_status == "loaded" and 
        llm_status == "configured"
    ) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        vector_db=vector_db_status,
        embedding_model=embedding_status,
        llm=llm_status
    )

@app.post(
    "/api/v1/documents",
    response_model=UploadResponse,
    responses={
        413: {"model": ErrorResponse},
        400: {"model": ErrorResponse}
    },
    tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF document, validates size and signature, deduplicates based on content hash,
    and runs PDF text parsing, semantic chunking, and vector indexing.
    """
    start_time = time.time()
    
    # Read raw content to compute size and signatures
    content = await file.read()
    file_size = len(content)
    
    # Reset read pointer
    await file.seek(0)
    
    # 1. Security Check: Size & magic bytes prefix check
    validate_pdf_file(content)
    
    # 2. Compute SHA-256 hash of document content for deduplication
    file_hash = hashlib.sha256(content).hexdigest()
    
    # 3. Deduplication Check
    existing_doc = vector_store.get_document_by_hash(file_hash)
    if existing_doc:
        total_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Deduplication triggered for file. Reusing document ID: {existing_doc['document_id']}",
            extra={"extra_fields": {
                "document_id": existing_doc["document_id"],
                "file_hash": file_hash,
                "cached": True
            }}
        )
        return UploadResponse(
            document_id=existing_doc["document_id"],
            filename=file.filename,
            upload_time=existing_doc["upload_time"],
            hash=file_hash,
            file_size=file_size,
            pages=existing_doc["pages"],
            chunks=existing_doc["chunks"],
            embedding_model=existing_doc["embedding_model"],
            processing_time_ms=total_time_ms,
            metrics=ProcessingMetrics(
                parse_time_ms=0,
                chunking_time_ms=0,
                embedding_time_ms=0,
                indexing_time_ms=0,
                total_time_ms=total_time_ms
            )
        )
        
    # 4. Process new document
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    token = document_id_ctx.set(doc_id)
    
    try:
        # Parse PDF
        parse_start = time.time()
        pages_data = parse_pdf(content)
        parse_time_ms = int((time.time() - parse_start) * 1000)
        
        # Chunk Text
        chunk_start = time.time()
        chunker = PageAwareChunker(
            target_tokens=settings.CHUNK_SIZE,
            overlap_tokens=settings.CHUNK_OVERLAP
        )
        chunks = chunker.chunk_document(pages_data)
        chunking_time_ms = int((time.time() - chunk_start) * 1000)
        
        # Generate Embeddings
        embed_start = time.time()
        chunk_texts = [c["chunk_text"] for c in chunks]
        embeddings = embedder.encode(chunk_texts)
        embedding_time_ms = int((time.time() - embed_start) * 1000)
        
        # Upsert Chunks into ChromaDB
        index_start = time.time()
        vector_store.upsert_document(
            document_id=doc_id,
            filename=file.filename,
            file_hash=file_hash,
            file_size=file_size,
            pages_count=len(pages_data),
            chunks=chunks,
            embeddings=embeddings
        )
        indexing_time_ms = int((time.time() - index_start) * 1000)
        
        total_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "New PDF uploaded and indexed successfully.",
            extra={"extra_fields": {
                "document_id": doc_id,
                "chunks_count": len(chunks),
                "pages_count": len(pages_data),
                "processing_time_ms": total_time_ms
            }}
        )
        
        # Pull saved meta (upload_time)
        meta = vector_store.get_document_by_id(doc_id)
        upload_time = meta["upload_time"] if meta else datetime.now(timezone.utc).isoformat()
        
        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            upload_time=upload_time,
            hash=file_hash,
            file_size=file_size,
            pages=len(pages_data),
            chunks=len(chunks),
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            processing_time_ms=total_time_ms,
            metrics=ProcessingMetrics(
                parse_time_ms=parse_time_ms,
                chunking_time_ms=chunking_time_ms,
                embedding_time_ms=embedding_time_ms,
                indexing_time_ms=indexing_time_ms,
                total_time_ms=total_time_ms
            )
        )
    finally:
        document_id_ctx.reset(token)

@app.get(
    "/api/v1/documents",
    response_model=List[DocumentResponse],
    tags=["Documents"]
)
async def list_documents():
    """Lists metadata for all documents registered in the system."""
    docs = vector_store.list_documents()
    return [DocumentResponse(**d) for d in docs]

@app.get(
    "/api/v1/documents/{document_id}",
    response_model=DocumentResponse,
    responses={
        404: {"model": ErrorResponse}
    },
    tags=["Documents"]
)
async def get_document_by_id(document_id: str):
    """Retrieves metadata of a single document by its ID."""
    doc = vector_store.get_document_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError()
    return DocumentResponse(**doc)

@app.post(
    "/api/v1/documents/{document_id}/query",
    response_model=StructuredQAResponse,
    responses={
        404: {"model": ErrorResponse}
    },
    tags=["Query"]
)
async def query_document(document_id: str, request_payload: QueryRequest):
    """Queries a document by ID using similarity vector search and an LLM provider."""
    token = document_id_ctx.set(document_id)
    try:
        # 1. Verify document exists
        doc = vector_store.get_document_by_id(document_id)
        if not doc:
            raise DocumentNotFoundError()
            
        start_time = time.time()
        
        # 2. Embed the user's query
        query_embedding = embedder.encode([request_payload.query])[0]
        
        # 3. Retrieve similar chunks matching the document ID
        retrieved_chunks = vector_store.query_similarity(
            document_id=document_id,
            query_embedding=query_embedding,
            top_k=settings.TOP_K
        )
        
        # 4. Generate structured Q&A response via LLM
        qa_response = llm_service.generate_answer(
            query=request_payload.query,
            retrieved_chunks=retrieved_chunks
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Query execution complete.",
            extra={"extra_fields": {
                "document_id": document_id,
                "latency_ms": latency_ms,
                "found_answer": qa_response.found,
                "confidence": qa_response.confidence.value
            }}
        )
        
        return qa_response
    finally:
        document_id_ctx.reset(token)
