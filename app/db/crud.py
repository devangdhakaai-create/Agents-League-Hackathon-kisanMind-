import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.db.models import FarmerSession, Advisory, ReasoningStep, ToolCache

logger = logging.getLogger(__name__)

def create_session(
        db: Session,
        *,
        crop: str,
        location_name: str,
        latitude: float,
        longitude: float,
        soil_type: str,
        farm_size_acres: float,
        free_text: Optional[str] = None,
) -> FarmerSession:
    session = FarmerSession(
        crop=crop,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        soil_type=soil_type,
        farm_size_acres=farm_size_acres,
        free_text=free_text,
        status="pending",
    )
    db.add(session)
    db.flush()
    logger.info(f"Created FarmerSession {session.id} for crop={crop}")
    return session

def get_session(db: Session, session_id: str) -> Optional[FarmerSession]:
    return db.query(FarmerSession).filter(
        FarmerSession.id == session_id
    ).first()

def get_recent_sessions(db: Session, limit: int = 10) -> list[FarmerSession]:
    return (
        db.query(FarmerSession)
        .order_by(desc(FarmerSession.created_at))
        .limit(limit)
        .all()
    )

def update_session_status(
    db: Session,
    session_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> Optional[FarmerSession]:
    session = get_session(db, session_id)
    if not session:
        logger.warning(f"Attempted to update non-existent session {session_id}")
        return None
    session.status = status
    if error_message:
        session.error_message = error_message
    db.flush()
    logger.info(f"Session {session_id} status → {status}")
    return session

def create_advisory(
    db: Session,
    *,
    session_id: str,
    recommendation: str,
    confidence: float,
    reasoning_text: str,
    risk_flags: list,
    actions: list,
    raw_llm_output: Optional[dict] = None,
    weather_summary: Optional[dict] = None,
    crop_summary: Optional[dict] = None,
    soil_summary: Optional[dict] = None,
    market_summary: Optional[dict] = None,
) -> Advisory:
    advisory = Advisory(
        session_id=session_id,
        recommendation=recommendation,
        confidence=confidence,
        reasoning_text=reasoning_text,
        risk_flags=risk_flags,
        actions=actions,
        raw_llm_output=raw_llm_output,
        weather_summary=weather_summary,
        crop_summary=crop_summary,
        soil_summary=soil_summary,
        market_summary=market_summary,
    )
    db.add(advisory)
    db.flush()
    logger.info(f"Created Advisory {advisory.id} for session {session_id} confidence={confidence:.2f}")
    return advisory

def get_advisory_by_session(db: Session, session_id: str) -> Optional[Advisory]:
    return db.query(Advisory).filter(Advisory.session_id == session_id).first()

def create_reasoning_step(
    db: Session,
    *,
    session_id: str,
    step_number: int,
    thought: str,
    tool_name: str,
    tool_args: dict,
    observation: Optional[dict] = None,
    is_final: bool = False,
    duration_ms: Optional[int] = None,
) -> ReasoningStep:
    step = ReasoningStep(
        session_id=session_id,
        step_number=step_number,
        thought=thought,
        tool_name=tool_name,
        tool_args=tool_args,
        observation=observation,
        is_final=is_final,
        duration_ms=duration_ms,
    )
    db.add(step)
    db.flush()
    logger.debug(f"ReasoningStep {step_number} persisted: session={session_id[:8]} tool={tool_name}")
    return step

def get_reasoning_trace(db: Session, session_id: str) -> list[ReasoningStep]:
    return (
        db.query(ReasoningStep)
        .filter(ReasoningStep.session_id == session_id)
        .order_by(ReasoningStep.step_number)
        .all()
    )

def get_step_count(db: Session, session_id: str) -> int:
    return db.query(ReasoningStep).filter(ReasoningStep.session_id == session_id).count()

def get_cached_tool_result(db: Session, cache_key: str) -> Optional[dict]:
    now = datetime.now(timezone.utc)
    entry = (
        db.query(ToolCache)
        .filter(ToolCache.cache_key == cache_key, ToolCache.expires_at > now)
        .first()
    )
    if entry:
        logger.debug(f"Cache HIT for key={cache_key}")
        return entry.result
    logger.debug(f"Cache MISS for key={cache_key}")
    return None

def set_cached_tool_result(
    db: Session,
    *,
    tool_name: str,
    cache_key: str,
    result: dict,
    ttl_seconds: int = 3600,
) -> ToolCache:
    db.query(ToolCache).filter(ToolCache.cache_key == cache_key).delete()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    entry = ToolCache(
        tool_name=tool_name,
        cache_key=cache_key,
        result=result,
        expires_at=expires_at,
    )
    db.add(entry)
    db.flush()
    logger.debug(f"Cached {tool_name} result: key={cache_key} ttl={ttl_seconds}s")
    return entry

def complete_session(
    db: Session,
    *,
    session_id: str,
    recommendation: str,
    confidence: float,
    reasoning_text: str,
    risk_flags: list,
    actions: list,
    raw_llm_output: Optional[dict] = None,
    weather_summary: Optional[dict] = None,
    crop_summary: Optional[dict] = None,
    soil_summary: Optional[dict] = None,
    market_summary: Optional[dict] = None,
) -> tuple[FarmerSession, Advisory]:
    try:
        session = update_session_status(db, session_id, "completed")
        if not session:
            raise ValueError(f"Session {session_id} not found")
        advisory = create_advisory(
            db,
            session_id=session_id,
            recommendation=recommendation,
            confidence=confidence,
            reasoning_text=reasoning_text,
            risk_flags=risk_flags,
            actions=actions,
            raw_llm_output=raw_llm_output,
            weather_summary=weather_summary,
            crop_summary=crop_summary,
            soil_summary=soil_summary,
            market_summary=market_summary,
        )
        db.commit()
        logger.info(f"Session {session_id} completed. Advisory {advisory.id} created.")
        return session, advisory
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to complete session {session_id}: {e}")
        raise

def fail_session(db: Session, *, session_id: str, error_message: str) -> Optional[FarmerSession]:
    try:
        session = update_session_status(db, session_id, status="failed", error_message=error_message)
        db.commit()
        logger.warning(f"Session {session_id} marked as failed: {error_message}")
        return session
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to mark session {session_id} as failed: {e}")
        raise