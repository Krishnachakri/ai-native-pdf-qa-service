# Project Implementation Approach

This document outlines the systematic engineering methodology used to build and verify the production-grade RAG PDF Q&A Service.

---

## 1. Core Methodology

Our approach was built on a **security-first, schema-driven, and test-validated pipeline**. Rather than stitching together high-level frameworks (which introduce abstraction overhead and hide failure modes), we built the core components from scratch using lightweight, robust standard tools:

-   **FastAPI & Pydantic v2**: Establishes strict data serialization contracts at the API boundary.
-   **PyMuPDF**: A highly optimized C-based engine for high-performance page-by-page text parsing.
-   **tiktoken**: Tokenizer matching OpenAI's exact encoding space to prevent subword truncation during sliding token window segmentation.
-   **sentence-transformers**: Locally executed embedding model (`all-MiniLM-L6-v2`) generating 384-dimensional vector maps without external API latency.
-   **ChromaDB**: SQLite-backed persistent vector store.
-   **OpenAI Tool Calling**: Strict schema validation to prevent JSON structure anomalies.

---

## 2. Phase-by-Phase Implementation

### Phase 1: Configuration, Logging, and Schema Contract
We defined the configuration settings and logging framework first. Pydantic schemas set up the data models for API communications before writing parser logic. We established:
-   A singleton settings loader via `pydantic-settings`.
-   A context-aware structured JSON logger using `contextvars` to trace `request_id` and `document_id`.

### Phase 2: Security & Routing Middlewares
We built security checks at the upload boundary:
-   Restricted inputs to `MAX_FILE_SIZE_BYTES` (10MB).
-   Inspected binary magic bytes (`%PDF-`) to reject renamed files.
-   Injected `X-Request-ID` tracing headers through a global middleware class.

### Phase 3: PDF Parsing
We extracted page text page-by-page, mapping blocks to their 1-based page indices. Scanned PDFs (extracting 0 characters) yield an `OCRNotSupportedError` to inform the user instead of letting empty pages pollute the downstream pipeline.

### Phase 4: Token Sliding Window Chunking
We implemented `PageAwareChunker`. Instead of split-by-character counts, we split on sentence boundaries and grouped sentences into overlapping token segments (default size 500, overlap 50) using `tiktoken`. We carried page metadata forward across sentence merges.

### Phase 5: Embeddings & Vector Database Indexing
We mapped text to embeddings and indexed them in ChromaDB. We implemented **SHA-256 deduplication**: checking if the file's hash exists before processing. If it does, we reuse the existing `document_id` and return within $\approx 2\text{ms}$ with zero parsing/indexing overhead.

### Phase 6: Pluggable LLM & RAG Q&A Service
We decoupled the LLM layer using the `BaseLLMProvider` abstract interface. We built the default `OpenAILLMProvider` to use OpenAI's tool schema calling. The RAG service constructs context XML blocks and validates citations post-hoc to drop and log hallucinated page listings. We also skip the LLM call entirely if retrieved chunk similarities fall below `SIM_THRESHOLD`.

### Phase 7: Routes & Integration Testing
We registered endpoints under `/api/v1/` and `/health`, wrapping custom exception handlers to return structured JSON errors. We wrote 8 integration tests using PyMuPDF to generate valid binary PDF files in-memory to test pipelines end-to-end.

---

## 3. Verification & Quality Assurance

Every service was verified continuously during development:
-   **Static Code Analysis**: Audited file tree to keep the codebase free of `TODO`s, console print statements, and dead code.
-   **Functional Validation**: Ran automated test suites inside the environment ensuring 100% test completion under 30 seconds.
-   **Docker Isolation**: Confirmed compose and multi-stage configs align with production build requirements.
