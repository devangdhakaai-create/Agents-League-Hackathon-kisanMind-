from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker  # SQLAlchemy ke async components/classes ko import kar rahe hain
from sqlalchemy.orm import DeclarativeBase  # DeclarativeBase ko import kar rahe hain jo ORM models ke liye base class hoti hai
from app.config import get_settings  # get_settings function ko import kar rahe hain jo configuration settings provide karta hai

settings = get_settings()  # settings object ko get_settings function se initialize kar rahe hain

engine = create_async_engine(
    settings.database_url,  # database URL settings se le rahe hain
    echo=settings.app_env == "development",  # agar environment development hai to SQL queries ko echo karenge
    pool_pre_ping=True,  # connection pool ke liye pre-ping enable kar rahe hain    
    pool_size=10,  # connection pool size set kar rahe hain
    max_overflow=20,  # maximum overflow connections set kar rahe hain
)
