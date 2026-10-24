import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "Autonomous Document Agent"
    VERSION: str = "1.0.0"
    API_V1_STR: str = ""
    
    # Groq LLM Configuration
    # Defaults to active Groq models; LLMManager dynamically discovers available models on your key
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")
    MODEL_TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 4096
    
    # Retry and Resilience Settings
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: float = 1.0
    RETRY_MAX_WAIT: float = 6.0
    
    # Self-Check / Reflection Settings
    ENABLE_REFLECTION: bool = True
    MIN_ACCEPTABLE_SCORE: int = 7
    MAX_REFLECTION_ROUNDS: int = 2
    
    # Document Output Directory
    OUTPUT_DIR: str = str(Path(__file__).resolve().parent.parent / "outputs")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure output directory exists
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
