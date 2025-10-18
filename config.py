from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ========================
    # Database Configuration
    # ========================
    mongodb_url: str               # Required from .env
    mongodb_database: str          # Required from .env
    redis_url: str                 # Required from .env
    redis_pass: str                # Required from .env

    # ========================
    # Qdrant Configuration
    # ========================
    qdrant_url: str                # Required from .env
    qdrant_api_key: Optional[str] = None  # Optional

    # ========================
    # OpenAI Configuration
    # ========================
    openai_api_key: str            # Required from .env

    # ========================
    # Application Configuration
    # ========================
    secret_key: str                # Required from .env
    algorithm: str = "HS256"       # Optional, default "HS256"
    access_token_expire_minutes: int = 30  # Optional

    # ========================
    # Development / Environment
    # ========================
    debug: bool = True             # Optional, default True
    environment: str = "development"  # Optional, default "development"

    class Config:
        # Path to your .env file
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
