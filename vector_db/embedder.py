import requests
import psutil
from typing import Optional

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_PRIMARY = "qllama/bge-m3:q8_0"
MODEL_FALLBACK = "qllama/bge-m3:q4_k_m"

# q8_0 will not be used if RAM is below the following threshold.
RAM_THRESHOLD_GB = 3.5


def select_embedding_model() -> str:
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if available_gb >= RAM_THRESHOLD_GB:
        print(f"[embedder] Available RAM: {available_gb:.2f} GB → using {MODEL_PRIMARY}")
        return MODEL_PRIMARY
    else:
        print(f"[embedder] Available RAM: {available_gb:.2f} GB (low) → using fallback {MODEL_FALLBACK}")
        return MODEL_FALLBACK


def get_embedding(text: str, model: Optional[str] = None) -> list[float]:
    if model is None:
        model = select_embedding_model()

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompts": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    except Exception as e:
        if model == MODEL_PRIMARY:
            print(f"[embedder] Primary model failed ({e}), retrying with fallback.")
            return get_embedding(text, model=MODEL_FALLBACK)
        raise RuntimeError(f"[embedder] Both models failed. Last error: {e}")


def get_embedding_dimensions(model: Optional[str] = None) -> int:
    if model is None:
        model = select_embedding_model()
    test_vector = get_embedding("dimension check", model=model)
    return len(test_vector)