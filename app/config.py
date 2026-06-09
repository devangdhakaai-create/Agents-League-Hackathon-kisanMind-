from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    GITHUB_TOKEN: str
    GITHUB_MODELS_BASE_URL: str = "https://models.inference.ai.azure.com"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.2
    MAX_REASONING_STEPS: int = 6
    LLM_TIMEOUT_SECONDS: int = 30
    DATABASE_URL: str
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1/forecast"
    WEATHER_FORCAST_DAYS: int = 7
    APP_NAME: str = "KisanMind"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Agricultural reasoning agent for smarter farming decisions"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: list[str] = ["*"]
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()