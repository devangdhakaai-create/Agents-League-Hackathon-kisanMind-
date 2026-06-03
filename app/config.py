from function import lru_cache
from pydentic_settings import BaseSettings, SettingConfigDict

class Settings(BaseSettings):
    model_config = SettingConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    app_env: str = "development"
    app_secret_key: str = "changeme"
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://kisanmind:password@localhost:5432/kisanmind_db"
    database_sync_url: str = "postgresql://kisanmind:password@localhost:5432/kisanmind_db"
azure_ai_project_connection_string: str = ""
azure_ai_agent_model: str = "gpt-4o"
azure_ai_agent_name: str = ""
azure_client_secret: str = ""
azure_tenant_id: str = ""

openweather_api_key: str = ""
openweather_base_url: str = "https://api.openweathermap.org/data/2.5"
mandi_api_key: str = ""
mandi_base_url: str = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d007"
allowed_origins: str = "http://localhost:3000,http://localhost:5173"

@property
def cors_origins(self) -> list[str]:
    return [o.strip() for o in self.allowed_origins.split(",")]

@property
def is_production(self) -> bool:
    return self.app_env == "production"

@lru_cache
def get_settings() -> Settings:
    return Settings