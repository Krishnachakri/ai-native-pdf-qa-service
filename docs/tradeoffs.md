# Architectural Trade-offs & Engineering Rationale

This document details the engineering trade-offs made in designing the RAG PDF Q&A backend, analyzing the pros, cons, and mitigation strategies for production environments.

---

## 1. Local CPU Embeddings vs. Cloud APIs

We chose to run the `all-MiniLM-L6-v2` embedding model locally via `sentence-transformers`.

### Pros
-   **No Network Latency/Failures**: Vector generation does not require external API calls, avoiding network jitter and downtime risk.
-   **Zero Incremental Cost**: No token fees or API charges for embedding generations.
-   **Strict Data Privacy**: Text chunks are embedded completely in-memory on our server, ensuring raw customer PDF data never leaks to third-party embedding providers.

### Cons
-   **CPU Thread Saturation**: Generating embeddings is highly compute-intensive. In a typical containerized environment (e.g., AWS ECS or GCP Cloud Run with 1–2 vCPUs), concurrent PDF uploads will saturate CPU cores, slowing down API request routing and blocking the FastAPI event loop.
-   **Container Bloat**: The runtime dependencies (PyTorch, numpy, transformers) increase the Docker image size by $\approx 350\text{MB}$ and require active RAM overhead.

### Mitigation
In a high-throughput production environment, we would offload embedding generation to a dedicated worker pool (using a task queue like Celery or Google Cloud Tasks) or switch to a managed cloud API (like OpenAI's embedding API) to keep the web application containers stateless and lightweight.

---

## 2. Stateless Scaling vs. In-Memory Rate Limiting

We explicitly removed local, in-memory IP-based rate limiting.

### Rationale
In modern production deployments, web applications run across multiple stateless server instances (e.g., Kubernetes pods, ECS tasks) behind a load balancer. 
-   **In-Memory State Lock-in**: An in-memory rate limiter relies on a local dictionary to track IP addresses. If a client's requests are distributed across 3 stateless tasks, the rate limiter cannot calculate their collective rate. This leads to inconsistent rate-limit enforcement.
-   **Memory Overhead**: Under high load, maintaining a registry of client IPs in-memory inside the Python process can lead to memory leakage.

### Production Solution
Rate limiting should be enforced at the **API Gateway/Ingress layer** (e.g., Cloudflare, Nginx, Kong) or centralized using a distributed cache (like Redis) running a leaky-bucket algorithm. This preserves application statelessness.

---

## 3. ChromaDB Persistent Client vs. Managed Vector Cluster

We utilized ChromaDB's `PersistentClient` pointing to a local directory (`./chroma_data`).

### Pros
-   **Operational Simplicity**: No external database cluster to deploy, configure, secure, and pay for. 
-   **Fast Queries**: Reads and writes go directly to local disk (via SQLite and local HNSW indices), bypassing network hops.

### Cons
-   **SQLite Write Lock Contention**: Chroma's persistent client writes to a local SQLite database file. SQLite does not support concurrent write transactions; concurrent document uploads will result in database locks and timeout failures.
-   **Stateless Scaling Barrier**: Since the vector database resides on local disk, we cannot scale our FastAPI instances horizontally without synchronizing the disks or using a shared network storage (which degrades HNSW file read performance).

### Production Solution
For a production deployment, we would decouple the vector store by switching the client connection to a managed vector database cluster (e.g. hosted Chroma, Qdrant, Pinecone, or Milvus) using HTTP/gRPC clients.

---

## 4. Regex-Based Sentence Chinking vs. Semantic Layout Parsing

Our chunker splits text into sentences using a regex lookahead and groups them into token windows.

### Pros
-   **Lightweight**: Avoids pulling heavy NLP packages (like NLTK or SpaCy) which slow down container startup times.
-   **Guarantees Completeness**: Ensures that we do not break text mid-sentence, preserving context structure.

### Cons
-   **Abbreviation Fails**: Technical abbreviations (like "e.g.", "i.e.", "Dr.") followed by spaces are misidentified as sentence ends, splitting clauses mid-thought.
-   **Punctuation-Free Text**: If a PDF contains lists, logs, or code blocks without periods, the regex fails to find boundaries, resulting in massive chunks that exceed the target size.

### Production Solution
Use a layout-aware PDF parser (like `Unstructured` or `Marker`) that extracts document sections, paragraphs, and tables as logical semantic units rather than raw lines of text.
