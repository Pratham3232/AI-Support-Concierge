from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-prod"

    database_url: str = "sqlite+aiosqlite:///./helix_srop.db"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "helix_docs"

    google_api_key: str = ""
    adk_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"

    llm_timeout_seconds: int = 30
    tool_timeout_seconds: int = 10

    score_threshold: float = 0.0  # filter retrieval below this; 0 disables


settings = Settings()
