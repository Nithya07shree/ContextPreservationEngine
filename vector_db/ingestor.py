"""
Orchestrates the ingestion pipeline:
  file -> chunker -> embedder -> ChromaDB upsert
"""

import os
from pathlib import Path
from typing import Optional

import chromadb

from chunker import chunk_code_file, chunk_slack_export, chunk_jira_csv
from embedder import get_embedding, select_embedding_model
from config import CHROMA_PATH, COLLECTION_NAME

# File extensions this pipeline will attempt to ingest as code
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".md", ".rst", ".cfg", ".ini",
}


def get_collection(chroma_path: str = None) -> chromadb.Collection:
    if chroma_path is None:
        chroma_path = CHROMA_PATH
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}, 
    )
    return collection


def ingest_code_file(
    file_path: str | Path,
    collection: chromadb.Collection,
    model: Optional[str] = None,
    project="unknown",
) -> int:
    chunks = chunk_code_file(file_path)
    for c in chunks:
        c["metadata"]["project"] = project
    return _upsert_chunks(chunks, collection, model)


def ingest_slack_file(
    file_path: str | Path,
    collection: chromadb.Collection,
    channel_name: str = "",
    model: Optional[str] = None,
    project="unknown",
) -> int:
    chunks = chunk_slack_export(file_path, channel_name=channel_name)
    for c in chunks:
        c["metadata"]["project"] = project
    return _upsert_chunks(chunks, collection, model)


def ingest_jira_file(
    file_path: str | Path,
    collection: chromadb.Collection,
    model: Optional[str] = None,
    project="unknown",
) -> int:
    chunks = chunk_jira_csv(file_path)
    for c in chunks:
        c["metadata"]["project"] = project
    return _upsert_chunks(chunks, collection, model)

# Batch / directory ingestion
def ingest_directory(
    directory: str | Path,
    collection: chromadb.Collection,
    model: Optional[str] = None,
    recursive: bool = True,
    project: str = "unknown",
) -> dict[str, int]:

    directory = Path(directory)
    results = {}
    pattern = "**/*" if recursive else "*"

    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        try:
            if ext in CODE_EXTENSIONS:
                count = ingest_code_file(file_path, collection, model, project=project)
                results[str(file_path)] = count
                print(f"[ingestor] code   {file_path.name}: {count} chunks")

            elif ext == ".json":
                count = ingest_slack_file(file_path, collection, model=model, project=project)
                results[str(file_path)] = count
                print(f"[ingestor] slack  {file_path.name}: {count} chunks")

            elif ext == ".csv":
                count = ingest_jira_file(file_path, collection, model, project=project)
                results[str(file_path)] = count
                print(f"[ingestor] jira   {file_path.name}: {count} chunks")

            else:
                print(f"[ingestor] skip   {file_path.name} (unsupported extension: {ext})")

        except Exception as e:
            print(f"[ingestor] ERROR  {file_path.name}: {e}")
            results[str(file_path)] = -1
    return results

# Core upsert logic
def _upsert_chunks(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: Optional[str] = None,
) -> int:
    if not chunks:
        return 0

    if model is None:
        model = select_embedding_model()

    BATCH_SIZE = 50
    total_upserted = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        batchid = [c["chunk_id"] for c in batch if c["content"].strip()]
        try:
            existing = set(collection.get(ids=batchid)["ids"])
        except Exception:
            existing = set()

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in batch:
            content = chunk["content"]
            if not content.strip():
                continue

            if chunk["chunk_id"] in existing:
                continue

            try:
                embedding = get_embedding(content, model=model)
            except Exception as e:
                print(f"[ingestor] Embedding failed for chunk {chunk['chunk_id']}: {e}")
                continue

            ids.append(chunk["chunk_id"])
            embeddings.append(embedding)
            documents.append(content)
            safe_metadata = {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in chunk["metadata"].items()
            }
            metadatas.append(safe_metadata)

        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total_upserted += len(ids)

    return total_upserted

# Similarity search
def search(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    source_type: Optional[str] = None,
    project: Optional[str] = None,
    model: Optional[str] = None,
) -> list[dict]:
    if model is None:
        model = select_embedding_model()

    query_embedding = get_embedding(query, model=model)

    filters = {}
    if source_type:
        filters["source_type"] = {"$eq": source_type}
    if project:
        filters["project"] = {"$eq": project}

    if len(filters) == 0:
        where = None
    elif len(filters) == 1:
        where = filters
    else:
        where = {"$and": [{k: v} for k, v in filters.items()]}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "content": doc,
            "metadata": meta,
            "distance": dist,
        })

    return output
