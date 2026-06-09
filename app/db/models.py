import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Text, ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.db.database import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

class FarmerSession(Base):
    __tablename__ = "farmer_sessions"
    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid, nullable=False)
    crop = Column(String(100), nullable=False)
    location_name = Column(String(200), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    soil_type = Column(String(50), nullable=False)
    farm_size_acres = Column(Float, nullable=False)
    free_text = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
    advisory = relationship("Advisory", back_populates="session", uselist=False, cascade="all, delete-orphan")
    reasoning_steps = relationship("ReasoningStep", back_populates="session", order_by="ReasoningStep.step_number", cascade="all, delete-orphan")
    def __repr__(self):
        return f"<FarmerSession id={self.id[:8]} crop={self.crop} status={self.status}>"

class Advisory(Base):
    __tablename__ = "advisories"
    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid, nullable=False)
    session_id = Column(UUID(as_uuid=False), ForeignKey("farmer_sessions.id", ondelete="CASCADE"), nullable=False, unique=True)
    recommendation = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    reasoning_text = Column(Text, nullable=False)
    risk_flags = Column(JSONB, nullable=False, default=list)
    actions = Column(JSONB, nullable=False, default=list)
    raw_llm_output = Column(JSONB, nullable=True)
    weather_summary = Column(JSONB, nullable=True)
    crop_summary = Column(JSONB, nullable=True)
    soil_summary = Column(JSONB, nullable=True)
    market_summary = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    session = relationship("FarmerSession", back_populates="advisory")
    def __repr__(self):
        return f"<Advisory id={self.id[:8]} confidence={self.confidence:.2f}>"

class ReasoningStep(Base):
    __tablename__ = "reasoning_steps"
    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid, nullable=False)
    session_id = Column(UUID(as_uuid=False), ForeignKey("farmer_sessions.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    thought = Column(Text, nullable=False)
    tool_name = Column(String(100), nullable=False)
    tool_args = Column(JSONB, nullable=False, default=dict)
    observation = Column(JSONB, nullable=True)
    is_final = Column(Boolean, nullable=False, default=False)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    session = relationship("FarmerSession", back_populates="reasoning_steps")
    __table_args__ = (Index("ix_reasoning_steps_session_step", "session_id", "step_number"),)
    def __repr__(self):
        return f"<ReasoningStep session={self.session_id[:8]} step={self.step_number} tool={self.tool_name}>"

class ToolCache(Base):
    __tablename__ = "tool_cache"
    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid, nullable=False)
    tool_name = Column(String(100), nullable=False)
    cache_key = Column(String(255), nullable=False, unique=True)
    result = Column(JSONB, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    __table_args__ = (Index("ix_tool_cache_key", "cache_key"),)
    def __repr__(self):
        return f"<ToolCache tool={self.tool_name} key={self.cache_key} expires={self.expires_at}>"