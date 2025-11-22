from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    database_url: str = "sqlite:///./text_analysis.db"
    secret_key: str = "your-secret-key-here-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    max_text_length: int = 100000
    batch_size: int = 100
    num_topics: int = 5
    passes: int = 10
    alpha: str = "auto"
    beta: str = "auto"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()