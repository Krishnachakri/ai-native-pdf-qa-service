import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    CHROMA_DATA_PATH: str = "./chroma_data"
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4
    SIM_THRESHOLD: float = 0.5
    PROMPT_VERSION: str = "v1.0"
    LLM_PROVIDER: str = "openai"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
