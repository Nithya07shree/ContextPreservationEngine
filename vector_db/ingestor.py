"""
Orchestrates the ingestion pipeline:
  file -> chunker -> embedder -> ChromaDB upsert
"""

import os
from pathlib import Path
from typing import Optional

import chromadb

from chunker import chunk_code_file, chunk_slack_export, chunk_jira_csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedder import get_embedding, get_embeddings_batch, select_embedding_model
from config import CHROMA_PATH, COLLECTION_NAME

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".php", ".md", ".rst", ".cfg",
    ".ini", ".tpl", ".smarty", ".sql", ".xml",
    ".html", ".htm", ".css", ".sh", ".bash",
}


def get_collection(collection_name: str = None, chroma_path: str = None) -> chromadb.Collection:
    if chroma_path is None:
        chroma_path = CHROMA_PATH
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name or COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )
    return collection


def ingest_code_file(
    file_path: str | Path,
    collection: chromadb.Collection,
    model: Optional[str] = None,
    project: str = "unknown",
) -> int:
    chunks = chunk_code_file(file_path)
    for chunk in chunks:
        chunk["metadata"]["project"] = project
    return _upsert_chunks(chunks, collection, model)


def ingest_slack_file(
    file_path: str | Path,
    collection: chromadb.Collection,
    channel_name: str = "",
    model: Optional[str] = None,
    project: str = "unknown",
) -> int:
    chunks = chunk_slack_export(file_path, channel_name=channel_name)
    for chunk in chunks:
        chunk["metadata"]["project"] = project
    return _upsert_chunks(chunks, collection, model)


def ingest_jira_file(
    file_path: str | Path,
    collection: chromadb.Collection,
    model: Optional[str] = None,
    project: str = "unknown",
) -> int:
    chunks = chunk_jira_csv(file_path)
    for chunk in chunks:
        chunk["metadata"]["project"] = project
    return _upsert_chunks(chunks, collection, model)


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


#UPSERT
EMBED_BATCH_SIZE = 50
MAX_WORKERS = 4


def _process_batch(
    batch: list[dict],
    collection: chromadb.Collection,
    model: str,
) -> int:

    batch = [c for c in batch if c["content"].strip()]
    if not batch:
        return 0

    batch_ids = [c["chunk_id"] for c in batch]
    try:
        existing = set(collection.get(ids=batch_ids)["ids"])
    except Exception:
        existing = set()

    new_chunks = [c for c in batch if c["chunk_id"] not in existing]
    if not new_chunks:
        return 0

    texts = [c["content"] for c in new_chunks]
    try:
        embeddings = get_embeddings_batch(texts, model=model)
    except Exception as e:
        print(f"[ingestor] Batch embedding failed: {e}")
        return 0

    if len(embeddings) != len(new_chunks):
        print(f"[ingestor] Embedding count mismatch: got {len(embeddings)}, expected {len(new_chunks)}")
        return 0

    # Step 4 — prepare and upsert
    ids, docs, metas = [], [], []
    for chunk, embedding in zip(new_chunks, embeddings):
        ids.append(chunk["chunk_id"])
        docs.append(chunk["content"])
        safe_meta = {
            k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
            for k, v in chunk["metadata"].items()
        }
        metas.append(safe_meta)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=metas,
    )
    return len(ids)


def _upsert_chunks(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: Optional[str] = None,
) -> int:

    if not chunks:
        return 0

    if model is None:
        model = select_embedding_model()
    # Split all chunks into fixed-size batches
    batches = [
        chunks[i:i + EMBED_BATCH_SIZE]
        for i in range(0, len(chunks), EMBED_BATCH_SIZE)
    ]

    total_upserted = 0

    # Process batches concurrently — each worker handles one batch independently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_process_batch, batch, collection, model): batch
            for batch in batches
        }
        for future in as_completed(futures):
            try:
                total_upserted += future.result()
            except Exception as e:
                print(f"[ingestor] Batch failed: {e}")

    return total_upserted


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def search(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    source_type: Optional[str] = None,
    project: Optional[str] = None,
    model: Optional[str] = None,
) -> list[dict]:
    """
    Perform similarity search against the collection.
    Optionally filter by source_type: "code" | "slack" | "jira"
    Optionally filter by project: "scrapy" | "your-other-repo" | etc.
    Both filters can be combined — ChromaDB uses $and automatically.
    Returns list of result dicts with content, metadata, and distance.
    """
    if model is None:
        model = select_embedding_model()

    query_embedding = get_embedding(query, model=model)

    # Build where clause — supports source_type, project, or both combined
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
        # ChromaDB requires $and when combining multiple conditions
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