# =============================================================================
# app/api/routes.py
# =============================================================================
# PURPOSE: All FastAPI route handlers. Thin orchestration layer —
# validates input, calls the reasoning engine, persists results,
# returns typed responses. No business logic lives here.
# =============================================================================

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import crud
from app.agent.engine import ReasoningEngine
from app.api.schemas import (
    AdvisoryRequest,
    AdvisoryResponse,
    AdvisoryListItem,
    CropListItem,
    RegionListItem,
    SoilTypeListItem,
    ErrorResponse,
    HealthResponse,
    ReasoningStepResponse,
    RiskFlag,
    ActionItem,
)
from app.tools.crop import list_supported_crops, list_supported_regions
from app.tools.soil import list_supported_soil_types
from app.config import settings
from app.db.database import verify_connection

logger = logging.getLogger(__name__)

# Single router — all routes registered here, mounted in main.py
router = APIRouter()


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness probe — required by Azure Container Apps",
)
def health_check():
    """
    Azure Container Apps calls this endpoint to verify the app is alive.
    Returns database connectivity status and app metadata.
    """
    try:
        verify_connection()  # runs SELECT 1 against PostgreSQL
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version=settings.APP_VERSION,
        database=db_status,
        llm_endpoint=settings.GITHUB_MODELS_BASE_URL,
        environment=settings.ENVIRONMENT,
    )


# =============================================================================
# CORE ADVISORY ENDPOINT
# =============================================================================

@router.post(
    "/advisory",
    response_model=AdvisoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Advisory"],
    summary="Run reasoning engine and generate farmer advisory",
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Reasoning engine error"},
    },
)
def create_advisory(
    request: AdvisoryRequest,
    db: Session = Depends(get_db),  # injected session — auto closed after response
):
    """
    Main endpoint. Accepts farm context, runs the ReAct reasoning loop
    across all four tools, persists results, returns full advisory.

    FLOW:
        1. Create FarmerSession row (pending)
        2. Run ReasoningEngine (4 tool calls + LLM reasoning)
        3. Persist Advisory + ReasoningSteps
        4. Return AdvisoryResponse with full trace
    """

    # ------------------------------------------------------------------
    # STEP 1: Create session row before reasoning starts
    # Ensures every request is logged even if the engine crashes
    # ------------------------------------------------------------------
    try:
        session = crud.create_session(
            db,
            crop=request.crop,
            location_name=request.location_name,
            latitude=request.latitude,
            longitude=request.longitude,
            soil_type=request.soil_type,
            farm_size_acres=request.farm_size_acres,
            free_text=request.free_text,
        )
        db.commit()
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "session_creation_failed", "message": str(e)},
        )

    session_id = session.id
    logger.info(f"Session {session_id} created for crop={request.crop}")

    # ------------------------------------------------------------------
    # STEP 2: Run the reasoning engine
    # Engine never raises — returns fallback advisory on any failure
    # ------------------------------------------------------------------
    try:
        engine = ReasoningEngine(session_id=session_id)
        result = engine.run(
            crop=request.crop,
            location_name=request.location_name,
            latitude=request.latitude,
            longitude=request.longitude,
            soil_type=request.soil_type,
            farm_size_acres=request.farm_size_acres,
            free_text=request.free_text,
        )
    except Exception as e:
        # Unexpected engine crash — mark session failed and return 500
        logger.error(f"Engine crashed for session {session_id}: {e}", exc_info=True)
        crud.fail_session(db, session_id=session_id, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "engine_failure", "message": "Reasoning engine encountered an error."},
        )

    # Unpack engine result
    advisory_data   = result["advisory"]
    reasoning_steps = result["reasoning_steps"]
    tool_summaries  = result["tool_summaries"]
    total_ms        = result["total_duration_ms"]

    # ------------------------------------------------------------------
    # STEP 3: Persist completed advisory to PostgreSQL
    # complete_session() atomically marks session completed + creates advisory
    # ------------------------------------------------------------------
    try:
        completed_session, advisory_row = crud.complete_session(
            db,
            session_id=session_id,
            recommendation=advisory_data.get("recommendation", ""),
            confidence=advisory_data.get("confidence", 0.0),
            reasoning_text=advisory_data.get("reasoning", ""),
            risk_flags=advisory_data.get("risk_flags", []),
            actions=advisory_data.get("actions", []),
            raw_llm_output=advisory_data,
            weather_summary=tool_summaries.get("weather"),
            crop_summary=tool_summaries.get("crop"),
            soil_summary=tool_summaries.get("soil"),
            market_summary=tool_summaries.get("market"),
        )
    except Exception as e:
        # Advisory generated but couldn't be persisted — still return it
        # Mark session as failed so it's visible in the history view
        logger.error(f"Failed to persist advisory for session {session_id}: {e}")
        crud.fail_session(db, session_id=session_id, error_message=str(e))
        advisory_id = None
        advisory_row = None
    else:
        advisory_id = advisory_row.id if advisory_row else None

    # ------------------------------------------------------------------
    # STEP 4: Build and return the response
    # ------------------------------------------------------------------
    return _build_advisory_response(
        session_id=session_id,
        advisory_id=advisory_id,
        advisory_data=advisory_data,
        reasoning_steps=reasoning_steps,
        tool_summaries=tool_summaries,
        total_ms=total_ms,
        advisory_row=advisory_row,
    )


# =============================================================================
# GET ADVISORY BY SESSION ID
# =============================================================================

@router.get(
    "/advisory/{session_id}",
    response_model=AdvisoryResponse,
    tags=["Advisory"],
    summary="Retrieve a completed advisory and its reasoning trace",
)
def get_advisory(session_id: str, db: Session = Depends(get_db)):
    """
    Fetches a previously generated advisory with its full reasoning trace.
    Used by the demo UI to reload past advisories from the history sidebar.
    """
    # Load session
    session = crud.get_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Session {session_id} not found."},
        )

    # Load advisory
    advisory_row = crud.get_advisory_by_session(db, session_id)
    if not advisory_row:
        # Session exists but advisory not ready yet (still pending or failed)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "advisory_not_ready",
                "message": f"Advisory for session {session_id} is {session.status}.",
                "status": session.status,
            },
        )

    # Load reasoning steps
    steps = crud.get_reasoning_trace(db, session_id)

    # Reconstruct advisory_data dict from the stored advisory row
    advisory_data = {
        "recommendation":  advisory_row.recommendation,
        "confidence":      advisory_row.confidence,
        "reasoning":       advisory_row.reasoning_text,
        "risk_flags":      advisory_row.risk_flags or [],
        "actions":         advisory_row.actions or [],
        "sowing_advice":   (advisory_row.raw_llm_output or {}).get("sowing_advice", ""),
        "irrigation_advice": (advisory_row.raw_llm_output or {}).get("irrigation_advice", ""),
        "market_advice":   (advisory_row.raw_llm_output or {}).get("market_advice", ""),
    }

    # Convert ORM step objects to dicts for the response builder
    steps_as_dicts = [
        {
            "step_number": s.step_number,
            "thought":     s.thought,
            "tool_name":   s.tool_name,
            "tool_args":   s.tool_args or {},
            "observation": s.observation,
            "duration_ms": s.duration_ms,
        }
        for s in steps
    ]

    tool_summaries = {
        "weather": advisory_row.weather_summary,
        "crop":    advisory_row.crop_summary,
        "soil":    advisory_row.soil_summary,
        "market":  advisory_row.market_summary,
    }

    return _build_advisory_response(
        session_id=session_id,
        advisory_id=advisory_row.id,
        advisory_data=advisory_data,
        reasoning_steps=steps_as_dicts,
        tool_summaries=tool_summaries,
        total_ms=None,
        advisory_row=advisory_row,
    )


# =============================================================================
# SESSION HISTORY
# =============================================================================

@router.get(
    "/sessions",
    response_model=list[AdvisoryListItem],
    tags=["Advisory"],
    summary="List recent advisory sessions for demo history sidebar",
)
def list_sessions(limit: int = 10, db: Session = Depends(get_db)):
    """
    Returns the most recent sessions for the demo history view.
    Limit capped at 50 to prevent accidental large queries.
    """
    limit = min(limit, 50)  # hard cap regardless of query param
    sessions = crud.get_recent_sessions(db, limit=limit)

    items = []
    for s in sessions:
        # Load advisory for each session to get recommendation and confidence
        advisory = crud.get_advisory_by_session(db, s.id)
        items.append(AdvisoryListItem(
            session_id=s.id,
            crop=s.crop,
            location_name=s.location_name,
            status=s.status,
            recommendation=advisory.recommendation if advisory else None,
            confidence=advisory.confidence if advisory else None,
            created_at=s.created_at,
        ))

    return items


# =============================================================================
# FORM DROPDOWN DATA ENDPOINTS
# =============================================================================

@router.get(
    "/crops",
    response_model=list[CropListItem],
    tags=["Reference Data"],
    summary="List supported crops for form dropdown",
)
def get_crops():
    """Returns all crops the reasoning engine supports."""
    return [CropListItem(**c) for c in list_supported_crops()]


@router.get(
    "/regions",
    response_model=list[RegionListItem],
    tags=["Reference Data"],
    summary="List supported regions for form dropdown",
)
def get_regions():
    """Returns all regions the reasoning engine supports."""
    return [RegionListItem(**r) for r in list_supported_regions()]


@router.get(
    "/soils",
    response_model=list[SoilTypeListItem],
    tags=["Reference Data"],
    summary="List supported soil types for form dropdown",
)
def get_soils():
    """Returns all soil types the reasoning engine supports."""
    return [SoilTypeListItem(**s) for s in list_supported_soil_types()]


# =============================================================================
# RESPONSE BUILDER
# =============================================================================

def _build_advisory_response(
    session_id: str,
    advisory_id: str | None,
    advisory_data: dict,
    reasoning_steps: list,
    tool_summaries: dict,
    total_ms: int | None,
    advisory_row,
) -> AdvisoryResponse:
    """
    Assembles AdvisoryResponse from engine output and DB row.
    Shared by create_advisory and get_advisory to ensure consistent shape.
    """

    # Convert raw risk_flags dicts to RiskFlag schema objects
    risk_flags = [
        RiskFlag(
            severity=f.get("severity", "medium"),
            type=f.get("type", "weather"),
            message=f.get("message", ""),
        )
        for f in advisory_data.get("risk_flags", [])
        if f.get("message")  # skip empty-message flags
    ]

    # Convert raw action dicts to ActionItem schema objects
    actions = [
        ActionItem(
            priority=a.get("priority", i + 1),
            action=a.get("action", ""),
            timeframe=a.get("timeframe", "as soon as possible"),
        )
        for i, a in enumerate(advisory_data.get("actions", []))
        if a.get("action")  # skip empty-action items
    ]

    # Convert reasoning step dicts to ReasoningStepResponse schema objects
    trace = [
        ReasoningStepResponse(
            step_number=s.get("step_number", i + 1),
            thought=s.get("thought", ""),
            tool_name=s.get("tool_name", ""),
            tool_args=s.get("tool_args", {}),
            observation=s.get("observation"),
            duration_ms=s.get("duration_ms"),
        )
        for i, s in enumerate(reasoning_steps)
    ]

    return AdvisoryResponse(
        session_id=session_id,
        advisory_id=advisory_id,
        status="completed" if advisory_id else "failed",
        recommendation=advisory_data.get("recommendation", ""),
        confidence=advisory_data.get("confidence", 0.0),
        reasoning=advisory_data.get("reasoning", ""),
        sowing_advice=advisory_data.get("sowing_advice", ""),
        irrigation_advice=advisory_data.get("irrigation_advice", ""),
        market_advice=advisory_data.get("market_advice", ""),
        risk_flags=risk_flags,
        actions=actions,
        reasoning_trace=trace,
        weather_summary=tool_summaries.get("weather"),
        crop_summary=tool_summaries.get("crop"),
        soil_summary=tool_summaries.get("soil"),
        market_summary=tool_summaries.get("market"),
        total_duration_ms=total_ms,
        created_at=advisory_row.created_at if advisory_row else None,
    )