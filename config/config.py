"""
All modules import from here
"""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:3b"
    embedding_model: str = "qllama/bge-m3:q8_0"
    
    chroma_collection_name: str = "context_engine"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    top_k_results: int = 5
    class Config:
        env_file= ".env"
        
settings = Settings()