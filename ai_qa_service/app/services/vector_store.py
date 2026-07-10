import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import chromadb
from app.core.config import settings

class VectorStore:
    """
    Interface for ChromaDB vector store operations.
    Handles indexing, querying (scoped by document ID), listing, and deduplication.
    """
    def __init__(self):
        # Initialize persistent storage path
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DATA_PATH)
        # Create or fetch collection using cosine distance metric
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Checks if a document with the matching SHA-256 hash is already indexed."""
        existing = self.collection.get(where={"file_hash": file_hash}, limit=1)
        if existing and existing["ids"]:
            meta = existing["metadatas"][0]
            return {
                "document_id": meta["document_id"],
                "filename": meta["filename"],
                "upload_time": meta.get("upload_time", ""),
                "hash": meta["file_hash"],
                "file_size": meta["file_size"],
                "pages": meta["page_count"],
                "chunks": meta.get("chunk_count", 1),
                "embedding_model": settings.EMBEDDING_MODEL_NAME
            }
        return None

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata of a document matching the document_id."""
        existing = self.collection.get(where={"document_id": document_id}, limit=1)
        if existing and existing["ids"]:
            meta = existing["metadatas"][0]
            return {
                "document_id": meta["document_id"],
                "filename": meta["filename"],
                "upload_time": meta.get("upload_time", ""),
                "hash": meta["file_hash"],
                "file_size": meta["file_size"],
                "pages": meta["page_count"],
                "chunks": meta.get("chunk_count", 1),
                "embedding_model": settings.EMBEDDING_MODEL_NAME
            }
        return None

    def list_documents(self) -> List[Dict[str, Any]]:
        """Lists all distinct documents present in the vector store."""
        records = self.collection.get(include=["metadatas"])
        if not records or not records["metadatas"]:
            return []
            
        seen_docs = {}
        for meta in records["metadatas"]:
            doc_id = meta["document_id"]
            if doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "upload_time": meta.get("upload_time", ""),
                    "hash": meta["file_hash"],
                    "file_size": meta["file_size"],
                    "pages": meta["page_count"],
                    "chunks": meta.get("chunk_count", 1),
                    "embedding_model": settings.EMBEDDING_MODEL_NAME
                }
        return list(seen_docs.values())

    def upsert_document(
        self,
        document_id: str,
        filename: str,
        file_hash: str,
        file_size: int,
        pages_count: int,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """Indexes all document chunks and embeddings into the collection."""
        ids = []
        metadatas = []
        documents = []
        
        chunk_count = len(chunks)
        upload_time = datetime.now(timezone.utc).isoformat()
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            ids.append(f"{document_id}_chunk_{idx}")
            documents.append(chunk["chunk_text"])
            
            # Serialize the list of pages spanned (Chroma metadatas only allow string, int, float, bool)
            pages_json = json.dumps(chunk["pages"])
            
            metadatas.append({
                "document_id": document_id,
                "filename": filename,
                "file_hash": file_hash,
                "file_size": file_size,
                "page_count": pages_count,
                "chunk_count": chunk_count,
                "pages": pages_json,
                "excerpt": chunk["excerpt"],
                "prompt_version": settings.PROMPT_VERSION,
                "upload_time": upload_time
            })
            
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query_similarity(self, document_id: str, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks matching the query embedding inside a specific document.
        
        Returns:
            list[dict]: List of matches containing chunk text, pages, excerpt, and similarity.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_id": document_id}
        )
        
        if not results or not results["ids"] or not results["ids"][0]:
            return []
            
        matched_chunks = []
        for idx in range(len(results["ids"][0])):
            meta = results["metadatas"][0][idx]
            text = results["documents"][0][idx]
            # Chroma distances metric: Cosine distance is returned (0 = same, 2 = opposite)
            dist = results["distances"][0][idx] if (results["distances"] and idx < len(results["distances"][0])) else 1.0
            
            # Convert Cosine Distance to Cosine Similarity
            similarity = 1.0 - dist
            
            matched_chunks.append({
                "chunk_text": text,
                "pages": json.loads(meta["pages"]),
                "excerpt": meta["excerpt"],
                "similarity": similarity
            })
            
        return matched_chunks
