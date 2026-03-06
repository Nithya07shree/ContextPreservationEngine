
import time
import yaml
import chromadb
from pathlib import Path
from ollama import Client as OllamaClient

from config.config import settings
from config.logger import get_logger

logger = get_logger(__name__)

_chroma_client = chromadb.HttpClient(
    host=settings.chroma_host,
    port=settings.chroma_port,
)
_collection = _chroma_client.get_collection(name=settings.chroma_collection_name)
_ollama     = OllamaClient(host=settings.ollama_base_url)

def _load_restrictions() -> dict[str, list[str]]:
    policy_path = Path("config/access_policy.yaml")
    with open(policy_path) as f:
        policy = yaml.safe_load(f)
    restrictions = policy.get("restrictions", {})
    logger.info("Access policy loaded, roles = %s", list(restrictions.keys()))
    return restrictions

_RESTRICTIONS: dict[str, list[str]] = _load_restrictions()


def _is_blocked(meta: dict, role: str) -> bool:
    if role == "admin":
        return False

    blocked_keywords = _RESTRICTIONS.get(role, [])
    if not blocked_keywords:
        return False
    meta_blob = " ".join(
        str(v) for v in meta.values() if v is not None
    ).lower()

    for keyword in blocked_keywords:
        if keyword.lower() in meta_blob:
            logger.debug(
                "Blocked | keyword='%s'  role=%s  meta='%s'",
                keyword, role, meta_blob[:100],
            )
            return True

    return False


def _embed_query(query: str, retries: int = 3, delay: float = 1.0) -> list[float]:
    """
    Embeds the user query via Ollama with exponential backoff retries.
    Ollama can time out on the first call if the model is still loading.
    """
    last_exc = None
    wait     = delay

    for attempt in range(1, retries + 2):  
        try:
            response = _ollama.embeddings(
                model  = settings.embedding_model,
                prompt = query,
            )
            return response["embedding"]

        except Exception as e:
            last_exc = e
            if attempt <= retries:
                logger.warning(
                    "Embedding attempt %d/%d failed, retrying in %.1fs — %s",
                    attempt, retries + 1, wait, e,
                )
                time.sleep(wait)
                wait *= 2 
            else:
                logger.error("All %d embedding attempts failed — %s", retries + 1, e)

    raise RuntimeError(f"Failed to embed query after {retries + 1} attempts: {last_exc}")

def retrieve_context(
    query:           str,
    user_role:       str   = "onboarding",
    n_results:       int   = None,
    score_threshold: float = 0.45,
) -> list[dict]:
    if not query or not query.strip():
        logger.warning("retrieve_context() called with empty query")
        return []

    if user_role not in _RESTRICTIONS and user_role != "admin":
        logger.warning("Unknown role '%s' — defaulting to 'onboarding'", user_role)
        user_role = "onboarding"

    top_k   = n_results or settings.top_k_results
    fetch_k = top_k * 3 

    logger.debug(
        "Retrieving context | role=%s  query='%s...'  fetch_k=%d",
        user_role, query[:60], fetch_k,
    )

    try:
        query_embedding = _embed_query(query)
    except RuntimeError as e:
        logger.error("Skipping retrieval — embedding failed: %s", e)
        return []

    try:
        results = _collection.query(
            query_embeddings = [query_embedding],
            n_results        = fetch_k,  
            include          = ["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error("ChromaDB query failed: %s", e)
        return []

    chunks  = []
    blocked = 0

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)

        if similarity < score_threshold:
            logger.debug(
                "Dropping chunk (score %.4f < threshold %.2f) | source=%s",
                similarity, score_threshold, meta.get("source_type", "?"),
            )
            continue

        if _is_blocked(meta, user_role):
            blocked += 1
            continue

        chunks.append({
            "text":          doc,
            "metadata":      meta,
            "similarity_score": similarity,
            "file_name":     meta.get("file_name"),
            "file_path":     meta.get("file_path"),
            "function_name": meta.get("function_name"),
            "language":      meta.get("language"),
            "source_type":   meta.get("source_type"),
            "project":       meta.get("project"),
            "start_line":    meta.get("start_line"),
            "end_line":      meta.get("end_line"),
            "chunk_index":   meta.get("chunk_index"),
            "total_chunks":  meta.get("total_chunks"),
        })

        if len(chunks) == top_k:
            break

    logger.info(
        "Retrieved %d chunk(s) | blocked=%d  role=%s  top_score=%s",
        len(chunks),
        blocked,
        user_role,
        chunks[0]["similarity_score"] if chunks else "N/A",
    )
    return chunks

def check_connection() -> bool:
    try:
        heartbeat = _chroma_client.heartbeat()
        doc_count = _collection.count()
        logger.info(
            "ChromaDB conn. verified | heartbeat=%s  collection='%s'  docs=%d",
            heartbeat, settings.chroma_collection_name, doc_count,
        )
        return True
    except Exception as e:
        logger.error("ChromaDB connection check failed: %s", e)
        return False
