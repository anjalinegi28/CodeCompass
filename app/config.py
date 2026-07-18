"""
Central configuration for CodeCompass.

Every other module reads settings from here instead of calling os.environ
directly, so there is exactly one place that knows about env vars.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider selection
    llm_provider: str = "openai"  # openai | gemini | claude
    openai_api_key: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    vector_store: str = "chroma"  # chroma | faiss
    chroma_persist_dir: str = ".codecompass/chroma_db"
    faiss_index_path: str = ".codecompass/faiss_index"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Eval gate
    eval_threshold: float = 0.70

    # MLflow (SQLite backend — newer MLflow versions put the plain folder-based
    # filestore into maintenance mode and refuse to write to it)
    mlflow_tracking_uri: str = "sqlite:///./.codecompass/mlflow.db"
    mlflow_experiment_name: str = "codecompass-eval"

    # File types the ingester will read as text. Anything else is skipped.
    allowed_extensions: tuple = (
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt",
        ".md", ".mdx", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml",
    )

    # Directories to always skip during ingestion
    excluded_dirs: tuple = (
        ".git", "node_modules", "venv", ".venv", "__pycache__",
        "dist", "build", ".next", ".codecompass", "mlruns",
        ".pytest_cache", "egg-info",
    )

    # Max file size (bytes) to ingest — skip huge generated/binary-ish files
    max_file_size_bytes: int = 5_000_000

    # Max total upload size (bytes) per request — enforced in app/api/main.py
    # against the incoming zip/file payload before it's written to disk.
    max_upload_size_bytes: int = 1_000_000_000  # 1 GB


settings = Settings()