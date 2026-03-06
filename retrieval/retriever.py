import chromadb
import time
from ollama import Client as OllamaClient
from config.config import settings
from config.logger import get_logger

logger = get_logger(__name__)

_chroma_client = chromadb.HttpClient(
    host = settings.chroma_host,
    port = settings.chroma_port,
)

_collection = _chroma_client.get_collection(name=settings.chroma_collection_name)
_ollama = OllamaClient(host=settings.ollama_base_url)

def _embed_query(query: str, retries: int = 3, delay: float = 1.0) -> list[float]:
    last_exc = None
    wait = delay
    for attempt in range(1,retries*2):
        try:
            response=_ollama.embeddings(model = settings.embedding_model,
                prompt = query)
            return response["embedding"]
        except Exception as e:
            last_exc = e
            if attempt<=retries:
                logger.warning("Embedding attempt %d/%d failed, retrying in %.1fs-%s",attempt, retries+1, wait, e)
                time.sleep(wait)
                wait*=2
            else:
                logger.error("All %d embedding attempts failed - %s",retries+1,e)
    raise RuntimeError(f"Failed to embed query after {retries+1} attempts: {last_exc}")

def retrieve_context(
    query:            str,
    n_results:        int   = None,
    source_filter:    str   = None,
    score_threshold:  float = 0.45,   # below this similarity score, chunk is not useful
) -> list[dict]:
    top_k = n_results or settings.top_k_results
    logger.debug("Retrieving context | query='%s...'  top_k=%d", query[:60], top_k)
    where = {"source": source_filter} if source_filter else None

    try:
        query_embedding = _embed_query(query)
    except RuntimeError as e:
        logger.error("Skipping retrieval because embedding failed: %s", e)
        return []
    try:
        results = _collection.query(
            query_embeddings = [query_embedding],
            n_results        = top_k,
            where            = where,
            include          = ["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error("ChromaDB query failed: %s", e)
        return []
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)

        if similarity < score_threshold:
            logger.debug(
                "Dropping chunk (score %.4f < threshold %.2f),  source=%s",
                similarity, score_threshold, meta.get("source", "?"),
            )
            continue

        chunks.append({
            "text": doc,
            "metadata": meta,
            "similarity_score": similarity,
        })

    if not chunks:
        logger.warning(
            "No chunks passed the score threshold (%.2f) for query: '%s...'",
            score_threshold, query[:60],
        )
    else:
        logger.info(
            "Retrieved %d chunk(s) | top score: %.4f | sources: %s",
            len(chunks),
            chunks[0]["similarity_score"],
            list({c["metadata"].get("source", "?") for c in chunks}),
        )

    return chunks

def check_connection() -> bool:
    try:
        heartbeat = _chroma_client.heartbeat()
        doc_count = _collection.count()
        logger.info(
            "ChromaDB conn. verified heartbeat=%s  collection='%s'  docs=%d",
            heartbeat, settings.CHROMA_COLLECTION_NAME, doc_count,
        )
        return True
    except Exception as e:
        logger.error("ChromaDB connection check failed: %s", e)
        return False