import io
import json
import hashlib
from unittest.mock import MagicMock
import pytest
import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from app.main import app, vector_store, llm_service
from app.core.config import settings
from app.models.schemas import StructuredQAResponse, Citation, ConfidenceLevel
from app.services.llm_provider import BaseLLMProvider

# Initialize FastAPI Test Client
client = TestClient(app)

# Helper function to generate mock PDFs dynamically
def generate_pdf_bytes(pages_text: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        # Insert text onto the page
        page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

@pytest.fixture(autouse=True)
def clean_database():
    """Fixture that resets the ChromaDB collection before and after each test run."""
    try:
        vector_store.client.delete_collection("pdf_documents")
    except Exception:
        pass
    vector_store.collection = vector_store.client.get_or_create_collection(
        name="pdf_documents",
        metadata={"hnsw:space": "cosine"}
    )
    yield
    try:
        vector_store.client.delete_collection("pdf_documents")
    except Exception:
        pass

class MockLLMProvider(BaseLLMProvider):
    """Deterministic Mock LLM Provider for integration testing."""
    def __init__(self, response: StructuredQAResponse):
        self.response = response
        self.calls = []

    def generate_response(self, prompt: str, system_prompt: str, response_schema) -> StructuredQAResponse:
        self.calls.append((prompt, system_prompt))
        return self.response

# ==================== 1. ANSWER FOUND TEST ====================
def test_answer_found():
    # 1. Generate and upload a valid PDF
    pdf_data = generate_pdf_bytes([
        "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.11. "
        "It is built on top of Starlette and Pydantic."
    ])
    
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("fastapi_intro.pdf", pdf_data, "application/pdf")}
    )
    assert upload_response.status_code == 200
    doc_id = upload_response.json()["document_id"]
    
    # 2. Configure mock LLM response
    mock_qa = StructuredQAResponse(
        answer="FastAPI is a Python web framework built on top of Starlette.",
        citations=[Citation(page_number=1, excerpt="FastAPI is a modern, fast (high-performance), web framework")],
        found=True,
        confidence=ConfidenceLevel.HIGH
    )
    test_provider = MockLLMProvider(mock_qa)
    
    # Temporarily inject the mock LLM provider into the LLM service
    original_provider = llm_service.provider
    llm_service.provider = test_provider
    
    try:
        # 3. Query the document
        query_payload = {"query": "What is FastAPI?"}
        query_response = client.post(
            f"/api/v1/documents/{doc_id}/query",
            json=query_payload
        )
        assert query_response.status_code == 200
        data = query_response.json()
        assert data["found"] is True
        assert "Starlette" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["citations"][0]["page_number"] == 1
        assert "fast (high-performance)" in data["citations"][0]["excerpt"]
    finally:
        llm_service.provider = original_provider

# ==================== 2. ANSWER NOT FOUND (THRESHOLD) TEST ====================
def test_answer_not_found_below_threshold():
    # 1. Upload a PDF with specific context
    pdf_data = generate_pdf_bytes(["The boiling point of nitrogen is 77 Kelvin."])
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("nitrogen.pdf", pdf_data, "application/pdf")}
    )
    doc_id = upload_response.json()["document_id"]
    
    # 2. Query something unrelated with a strict similarity threshold
    original_threshold = settings.SIM_THRESHOLD
    settings.SIM_THRESHOLD = 0.9  # Set high similarity threshold to guarantee it fails
    
    test_provider = MockLLMProvider(StructuredQAResponse(answer="", citations=[], found=False, confidence=ConfidenceLevel.NONE))
    original_provider = llm_service.provider
    llm_service.provider = test_provider
    
    try:
        query_payload = {"query": "What is the capital of France?"}
        query_response = client.post(
            f"/api/v1/documents/{doc_id}/query",
            json=query_payload
        )
        
        assert query_response.status_code == 200
        data = query_response.json()
        # Assert LLM call was skipped (no calls recorded on mock) and answer is marked not found
        assert len(test_provider.calls) == 0
        assert data["found"] is False
        assert data["confidence"] == "NONE"
        assert "not found" in data["answer"].lower()
    finally:
        settings.SIM_THRESHOLD = original_threshold
        llm_service.provider = original_provider

# ==================== 3. SCANNED OR EMPTY PDF TEST ====================
def test_scanned_or_empty_pdf():
    # 1. Generate an empty PDF page (0 extractable characters)
    doc = fitz.open()
    doc.new_page()  # Blank page
    empty_pdf_bytes = doc.write()
    doc.close()
    
    # 2. Upload and assert 400 error return
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("scanned_blank.pdf", empty_pdf_bytes, "application/pdf")}
    )
    assert upload_response.status_code == 400
    data = upload_response.json()
    assert data["error"] == "OCR_NOT_SUPPORTED"
    assert "OCR is not supported" in data["message"]

# ==================== 4. MULTI-PAGE BOUNDARY CHUNKS TEST ====================
def test_multipage_boundary_chunks():
    # 1. Create a PDF where a sentence spans across pages
    pdf_data = generate_pdf_bytes([
        "This is page one. Sentence starts here and runs into page two text",  # Page 1
        "which continues the idea with more context on the second page."       # Page 2
    ])
    
    # 2. Set chunk parameters small enough to force merging across pages
    original_chunk_size = settings.CHUNK_SIZE
    settings.CHUNK_SIZE = 40  # Small token chunk size
    
    try:
        upload_response = client.post(
            "/api/v1/documents",
            files={"file": ("multipage.pdf", pdf_data, "application/pdf")}
        )
        assert upload_response.status_code == 200
        data = upload_response.json()
        assert data["pages"] == 2
        assert data["chunks"] > 0
        
        # Verify that Chroma stores a chunk listing pages [1, 2]
        records = vector_store.collection.get(where={"document_id": data["document_id"]}, include=["metadatas"])
        spanned_multipage = False
        for meta in records["metadatas"]:
            pages = json.loads(meta["pages"])
            if 1 in pages and 2 in pages:
                spanned_multipage = True
                break
        assert spanned_multipage is True, "No chunk was found spanning both page 1 and page 2."
    finally:
        settings.CHUNK_SIZE = original_chunk_size

# ==================== 5. LARGE PDF / SECURITY BOUNDARIES TEST ====================
def test_large_pdf_and_security_limits():
    # Test Size Boundary Enforcement
    original_max_size = settings.MAX_FILE_SIZE_BYTES
    settings.MAX_FILE_SIZE_BYTES = 500  # Set limit very low (500 bytes)
    
    try:
        # Create a tiny valid PDF (still > 500 bytes)
        pdf_data = generate_pdf_bytes(["Test size limits."])
        assert len(pdf_data) > 500
        
        upload_response = client.post(
            "/api/v1/documents",
            files={"file": ("size_test.pdf", pdf_data, "application/pdf")}
        )
        assert upload_response.status_code == 413
        data = upload_response.json()
        assert data["error"] == "FILE_TOO_LARGE"
        assert "exceeds the maximum" in data["message"]
    finally:
        settings.MAX_FILE_SIZE_BYTES = original_max_size
        
    # Test Magic Bytes Signature Enforcement
    fake_pdf_data = b"This is a text file renamed to pdf.pdf but doesn't have magic bytes signature."
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": ("bad_sig.pdf", fake_pdf_data, "application/pdf")}
    )
    assert upload_response.status_code == 400
    data = upload_response.json()
    assert data["error"] == "INVALID_PDF"
    assert "magic bytes signature mismatch" in data["message"]

# ==================== 6. DUPLICATE UPLOAD DEDUPLICATION TEST ====================
def test_duplicate_upload_deduplication():
    # 1. Create a valid PDF
    pdf_data = generate_pdf_bytes(["Deduplication test data is here."])
    
    # 2. Upload the first time
    response_1 = client.post(
        "/api/v1/documents",
        files={"file": ("dedup_test.pdf", pdf_data, "application/pdf")}
    )
    assert response_1.status_code == 200
    data_1 = response_1.json()
    doc_id_1 = data_1["document_id"]
    
    # 3. Upload the second time (exact duplicate)
    response_2 = client.post(
        "/api/v1/documents",
        files={"file": ("dedup_test.pdf", pdf_data, "application/pdf")}
    )
    assert response_2.status_code == 200
    data_2 = response_2.json()
    doc_id_2 = data_2["document_id"]
    
    # 4. Verify identical document ID (deduplication)
    assert doc_id_1 == doc_id_2
    
    # 5. Verify metrics show bypassed pipeline (parse, chunk, embed, and index are 0ms)
    metrics = data_2["metrics"]
    assert metrics["parse_time_ms"] == 0
    assert metrics["chunking_time_ms"] == 0
    assert metrics["embedding_time_ms"] == 0
    assert metrics["indexing_time_ms"] == 0

# ==================== 7. CORRUPTED PDF TEST ====================
def test_corrupted_pdf_parse_error():
    # starts with valid magic bytes but has junk content that fitz cannot open
    junk_pdf_data = b"%PDF-1.4\nthis is complete garbage content that cannot be parsed as a PDF"
    
    response = client.post(
        "/api/v1/documents",
        files={"file": ("corrupted.pdf", junk_pdf_data, "application/pdf")}
    )
    # ValueError caught and returned as structured HTTP 400
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "INVALID_PDF"
    assert "Failed to open PDF file" in data["message"]

# ==================== 8. MISSING API KEY TEST ====================
def test_missing_api_key_health_and_query():
    # 1. Set OPENAI_API_KEY to empty
    original_key = settings.OPENAI_API_KEY
    settings.OPENAI_API_KEY = ""
    
    try:
        # 2. Check that health endpoint reports misconfigured
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["llm"] == "misconfigured"
        assert health_data["status"] == "unhealthy"
        
        # 3. Verify that trying to run a query (where similarity threshold passes) returns the graceful fallback
        # Let's add a document first while API key is empty
        pdf_data = generate_pdf_bytes(["We need some query text context."])
        upload_response = client.post(
            "/api/v1/documents",
            files={"file": ("query_test.pdf", pdf_data, "application/pdf")}
        )
        doc_id = upload_response.json()["document_id"]
        
        # Running the query should trigger LLM call which will fail because key is missing,
        # but our llm_service will catch the exception and return a graceful internal LLM error answer.
        query_payload = {"query": "What is the query text?"}
        query_response = client.post(
            f"/api/v1/documents/{doc_id}/query",
            json=query_payload
        )
        assert query_response.status_code == 200
        query_data = query_response.json()
        assert query_data["found"] is False
        assert "Failed to generate an answer" in query_data["answer"]
    finally:
        settings.OPENAI_API_KEY = original_key

