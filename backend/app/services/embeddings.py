import os
import logging
from fastapi import Depends
import httpx
import asyncio
from typing import List, Dict, Optional, Any
from uuid import UUID
from datetime import datetime

logger = logging.getLogger(__name__)

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector

from app.models.document import Document, DocumentChunk
from app.models.memory import SemanticMemory
from app.models.project import RepositoryFile, Repository
from app.core.di import get_db_session
from app.core.config import settings
from app.core.reliability import CircuitBreaker

_emb_breaker = CircuitBreaker(failure_threshold=5, window_s=60, recovery_timeout=30)


class EmbeddingService:
    """Service for generating embeddings and storing in pgvector."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.ollama_base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.embedding_model = getattr(settings, "EMBEDDING_MODEL", "nomic-embed-text")
        self.embedding_dim = 768  # nomic-embed-text via Ollama only — DB Vector(768)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with CircuitBreaker — fast-fail when Ollama down."""
        if not texts:
            return []
        if not _emb_breaker.allow_request():
            logger.warning("Embedding circuit breaker %s, skipping %d texts", _emb_breaker.state, len(texts))
            return [[0.0] * self.embedding_dim for _ in texts]

        embeddings: List[List[float]] = []
        success = 0
        failed = 0
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                try:
                    response = await client.post(
                        f"{self.ollama_base_url}/api/embeddings",
                        json={"model": self.embedding_model, "prompt": text[:8000]},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        emb = data.get("embedding", [])
                        if emb and len(emb) == self.embedding_dim:
                            embeddings.append(emb)
                            success += 1
                        else:
                            embeddings.append([0.0] * self.embedding_dim)
                            failed += 1
                    else:
                        embeddings.append([0.0] * self.embedding_dim)
                        failed += 1
                except Exception as e:
                    logger.warning("Embedding generation failed: %s", e)
                    embeddings.append([0.0] * self.embedding_dim)
                    failed += 1

        if success > 0:
            _emb_breaker.record_success()
        if failed > 0 and failed == len(texts):
            _emb_breaker.record_failure()
        return embeddings

    async def embed_and_store_document_chunks(self, document_id: UUID) -> Dict:
        """Generate embeddings for all chunks of a document."""
        # Get chunks without embeddings
        result = await self.db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .where(DocumentChunk.embedding.is_(None))
        )
        chunks = result.scalars().all()

        if not chunks:
            return {"status": "completed", "chunks_processed": 0}

        texts = [chunk.content for chunk in chunks]
        embeddings = await self.generate_embeddings(texts)

        updated = 0
        for chunk, embedding in zip(chunks, embeddings):
            if embedding and len(embedding) == self.embedding_dim:
                chunk.embedding = embedding
                updated += 1

        await self.db.commit()

        return {
            "status": "completed",
            "chunks_processed": updated,
            "total_chunks": len(chunks),
        }

    async def embed_and_store_repository_files(self, repository_id: UUID) -> Dict:
        """Generate embeddings for all files in a repository."""
        result = await self.db.execute(
            select(RepositoryFile)
            .where(RepositoryFile.repository_id == repository_id)
            .where(RepositoryFile.content.is_not(None))
        )
        files = result.scalars().all()

        if not files:
            return {"status": "completed", "files_processed": 0}

        # Chunk large files
        all_chunks = []
        file_chunk_map = []

        for file in files:
            content = file.content
            if not content:
                continue

            # Simple chunking: split by lines, ~1000 chars per chunk
            chunk_size = 1000
            overlap = 200
            for i in range(0, len(content), chunk_size - overlap):
                chunk_text = content[i:i + chunk_size]
                if len(chunk_text.strip()) > 50:
                    all_chunks.append(chunk_text)
                    file_chunk_map.append(file.id)

        if not all_chunks:
            return {"status": "completed", "files_processed": 0}

        # Generate embeddings
        embeddings = await self.generate_embeddings(all_chunks)

        # Store as document chunks linked to files
        doc_result = await self.db.execute(
            select(Document).where(Document.project_id.in_(
                select(Repository.project_id).where(Repository.id == repository_id)
            )).limit(1)
        )
        project_doc = doc_result.scalar_one_or_none()

        if not project_doc:
            # Get project from repository
            repo_result = await self.db.execute(
                select(Repository).where(Repository.id == repository_id)
            )
            repo = repo_result.scalar_one_or_none()
            if repo:
                project_doc = Document(
                    project_id=repo.project_id,
                    title=f"Repository: {repo.name}",
                    source_type="repository",
                    source_path=repo.url,
                    doc_metadata={"repository_id": str(repository_id)},
                )
                self.db.add(project_doc)
                await self.db.commit()
                await self.db.refresh(project_doc)

        # Save chunks with embeddings
        saved = 0
        for i, (chunk_text, embedding, file_id) in enumerate(zip(all_chunks, embeddings, file_chunk_map)):
            if embedding and len(embedding) == self.embedding_dim:
                chunk = DocumentChunk(
                    document_id=project_doc.id,
                    file_id=file_id,
                    chunk_index=i,
                    content=chunk_text,
                    token_count=len(chunk_text) // 4,
                    embedding=embedding,
                    chunk_metadata={"repository_file_id": str(file_id)},
                )
                self.db.add(chunk)
                saved += 1

        await self.db.commit()

        return {
            "status": "completed",
            "files_processed": len(files),
            "chunks_created": saved,
        }

    async def semantic_search(
        self,
        query: str,
        project_id: Optional[UUID] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        table_name: str = "document_chunks",
    ) -> List[Dict]:
        """Perform semantic search using pgvector.

        Args:
            table_name: "document_chunks" (default, JOIN with documents) or "semantic_memory"
        """
        query_embedding = (await self.generate_embeddings([query]))[0]

        if not query_embedding or len(query_embedding) != self.embedding_dim:
            return []

        emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        if table_name == "semantic_memory":
            base_query = f"""
                SELECT id, text, metadata,
                       1 - (embedding <=> '{emb_str}'::vector) as similarity
                FROM semantic_memory
                WHERE embedding IS NOT NULL
                    AND 1 - (embedding <=> '{emb_str}'::vector) >= :threshold
                ORDER BY embedding <=> '{emb_str}'::vector
                LIMIT :limit
            """
            result = await self.db.execute(text(base_query), {"threshold": similarity_threshold, "limit": limit})
            rows = result.fetchall()
            return [
                {
                    "id": str(row.id),
                    "text": row.text,
                    "metadata": row.metadata,
                    "similarity": float(row.similarity),
                }
                for row in rows
            ]

        base_query = f"""
            SELECT
                dc.id, dc.content, dc.chunk_metadata, dc.chunk_index, dc.token_count,
                d.id as document_id, d.title as document_title,
                d.source_type, d.doc_metadata as document_metadata,
                1 - (dc.embedding <=> '{emb_str}'::vector) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL
        """

        params: dict = {"limit": limit}

        if project_id:
            base_query += " AND d.project_id = :project_id"
            params["project_id"] = str(project_id)

        base_query += f"""
            AND 1 - (dc.embedding <=> '{emb_str}'::vector) >= :threshold
            ORDER BY dc.embedding <=> '{emb_str}'::vector
            LIMIT :limit
        """
        params["threshold"] = similarity_threshold

        result = await self.db.execute(text(base_query), params)
        rows = result.fetchall()

        return [
            {
                "chunk_id": str(row.id),
                "content": row.content,
                "metadata": row.chunk_metadata,
                "chunk_index": row.chunk_index,
                "token_count": row.token_count,
                "document_id": str(row.document_id),
                "document_title": row.document_title,
                "source_type": row.source_type,
                "document_metadata": row.document_metadata,
                "similarity": float(row.similarity),
            }
            for row in rows
        ]

    async def hybrid_search(
        self,
        query: str,
        project_id: Optional[UUID] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Hybrid search combining semantic and keyword search."""
        semantic_results = await self.semantic_search(query, project_id, limit, similarity_threshold=settings.SIMILARITY_THRESHOLD)

        result = await self.db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.content.ilike(f"%{query}%"))
            .limit(limit)
        )
        keyword_rows = result.scalars().all()

        combined: Dict[str, Dict] = {}
        for r in semantic_results:
            combined[r["chunk_id"]] = r

        for r in keyword_rows:
            rid = str(r.id)
            if rid not in combined:
                combined[rid] = {
                    "chunk_id": rid,
                    "content": r.content,
                    "chunk_index": r.chunk_index,
                    "token_count": r.token_count,
                    "similarity": 0.5,
                }

        ranked = sorted(combined.values(), key=lambda x: x.get("similarity", 0), reverse=True)[:limit]
        return ranked


async def get_embedding_service(db: AsyncSession = Depends(get_db_session)) -> EmbeddingService:
    return EmbeddingService(db)