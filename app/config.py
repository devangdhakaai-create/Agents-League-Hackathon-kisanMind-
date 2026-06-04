from function import lru_cache  # lru_cache ko import kar rahe hain jo caching ke liye use hota hai
from pydentic_settings import BaseSettings, SettingConfigDict  # BaseSettings aur SettingConfigDict ko import kar rahe hain jo env-based config handle karte hain

class Settings(BaseSettings):  # Settings class banayi jo BaseSettings se inherit karti hai
    model_config = SettingConfigDict(  # model_config define kar rahe hain jisme env file aur settings ka config hai
        env_file=".env",  # env file ka naam specify kar rahe hain
        env_file_encoding="utf-8",  # env file encoding UTF-8 rakhi hai
        case_sensitive=False,  # env variables case sensitive nahi honge
        extra="ignore",  # extra env variables ignore kiye jayenge
    )
    app_env: str = "development"  # default environment development rakha hai
    app_secret_key: str = "changeme"  # secret key default value "changeme" hai
    log_level: str = "INFO"  # logging level INFO rakha hai
    database_url: str = "postgresql+asyncpg://kisanmind:password@localhost:5432/kisanmind_db"  # async database connection URL
    database_sync_url: str = "postgresql://kisanmind:password@localhost:5432/kisanmind_db"  # sync database connection URL

azure_ai_project_connection_string: str = ""  # Azure AI project connection string empty rakha hai
azure_ai_agent_model: str = "gpt-4o"  # Azure AI agent model gpt-4o rakha hai
azure_ai_agent_name: str = ""  # Azure AI agent ka naam empty hai
azure_client_secret: str = ""  # Azure client secret empty rakha hai
azure_tenant_id: str = ""  # Azure tenant ID empty rakha hai

openweather_api_key: str = ""  # OpenWeather API key empty rakha hai
openweather_base_url: str = "https://api.openweathermap.org/data/2.5"  # OpenWeather API base URL
mandi_api_key: str = ""  # Mandi API key empty rakha hai
mandi_base_url: str = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d007"  # Mandi API base URL
allowed_origins: str = "http://localhost:3000,http://localhost:5173"  # Allowed CORS origins list

@property
def cors_origins(self) -> list[str]:  # property method jo allowed_origins ko list mein convert karta hai
    return [o.strip() for o in self.allowed_origins.split(",")]  # comma se split karke strip karte hain aur list return karte hain

@property
def is_production(self) -> bool:  # property method jo check karta hai ki environment production hai ya nahi
    return self.app_env == "production"  # agar app_env "production" hai to True return karega

@lru_cache
def get_settings() -> Settings:  # function jo Settings object ko cache karke return karta hai
    return Settings  # Settings class return karta hai
