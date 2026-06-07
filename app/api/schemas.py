# =============================================================================
# app/api/schemas.py
# =============================================================================
# PURPOSE: Pydantic models for all API request and response shapes.
# FastAPI uses these for automatic validation, serialisation, and
# OpenAPI docs generation. Every route handler uses types from this file.
# =============================================================================

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


# =============================================================================
# REQUEST SCHEMAS — what the farmer submits via the form
# =============================================================================

class AdvisoryRequest(BaseModel):
    """
    Input schema for POST /api/v1/advisory.
    All fields validated before the reasoning engine runs.
    """

    crop: str = Field(
        ...,                              # required field
        min_length=2,
        max_length=50,
        description="Crop identifier e.g. wheat, rice, cotton, maize, tomato, soybean",
        examples=["wheat"]
    )

    location_name: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Human-readable location name e.g. Pune, Maharashtra",
        examples=["Pune, Maharashtra"]
    )

    latitude: float = Field(
        ...,
        ge=-90.0,   # greater than or equal — geographic bounds
        le=90.0,
        description="Latitude of farm location",
        examples=[18.5204]
    )

    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude of farm location",
        examples=[73.8567]
    )

    soil_type: str = Field(
        ...,
        description="Soil type identifier e.g. black_cotton, loamy, sandy, clay, red_laterite, sandy_loam",
        examples=["black_cotton"]
    )

    farm_size_acres: float = Field(
        ...,
        gt=0.0,     # greater than — farm must have positive area
        le=10000.0, # upper bound — sanity check
        description="Farm size in acres",
        examples=[5.0]
    )

    region: str = Field(
        ...,
        description="Region identifier e.g. north_india, south_india, west_india, east_india, central_india",
        examples=["west_india"]
    )

    free_text: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional additional context from the farmer",
        examples=["Soil was waterlogged last month"]
    )

    @field_validator("crop")
    @classmethod
    def normalise_crop(cls, v: str) -> str:
        # Lowercase and strip whitespace — "Wheat" → "wheat"
        return v.lower().strip()

    @field_validator("soil_type")
    @classmethod
    def normalise_soil(cls, v: str) -> str:
        # Normalise soil type — "Black Cotton" → "black_cotton"
        return v.lower().strip().replace(" ", "_")

    @field_validator("region")
    @classmethod
    def normalise_region(cls, v: str) -> str:
        # Normalise region — "North India" → "north_india"
        return v.lower().strip().replace(" ", "_")


# =============================================================================
# RESPONSE COMPONENT SCHEMAS
# =============================================================================

class RiskFlag(BaseModel):
    """One risk warning in the advisory response."""
    severity: str   # critical | high | medium | low
    type: str       # weather | soil | market | timing | pest | water
    message: str    # human-readable description of the risk


class ActionItem(BaseModel):
    """One recommended action in the advisory response."""
    priority: int   # 1 = most urgent
    action: str     # what the farmer should do
    timeframe: str  # when to do it e.g. "within 3 days"


class ReasoningStepResponse(BaseModel):
    """One step of the agent's reasoning trace for the demo UI."""
    step_number: int
    thought: str        # what the agent was thinking
    tool_name: str      # which tool was called
    tool_args: dict     # arguments passed to the tool
    observation: Optional[dict] = None  # what the tool returned
    duration_ms: Optional[int] = None   # how long the tool call took


class WeatherSummary(BaseModel):
    """Compact weather summary stored with the advisory."""
    avg_temp_c: Optional[float] = None
    total_rainfall_mm: Optional[float] = None
    max_humidity_pct: Optional[float] = None
    drought_risk: Optional[bool] = None
    waterlogging_risk: Optional[bool] = None
    irrigation_needed: Optional[bool] = None
    irrigation_urgency: Optional[str] = None


class CropSummary(BaseModel):
    """Compact crop summary stored with the advisory."""
    crop: Optional[str] = None
    category: Optional[str] = None
    sowing_status: Optional[str] = None
    days_until_close: Optional[int] = None
    days_until_open: Optional[int] = None
    base_irrigation_days: Optional[int] = None
    drought_tolerance: Optional[str] = None
    waterlogging_tolerance: Optional[str] = None


class SoilSummary(BaseModel):
    """Compact soil summary stored with the advisory."""
    soil_type: Optional[str] = None
    irrigation_multiplier: Optional[float] = None
    adjusted_interval_days: Optional[int] = None
    waterlogging_risk: Optional[str] = None
    drainage_action: Optional[bool] = None
    fertility: Optional[str] = None


class MarketSummary(BaseModel):
    """Compact market summary stored with the advisory."""
    signal: Optional[str] = None
    confidence: Optional[float] = None
    current_price: Optional[float] = None
    msp_position: Optional[str] = None
    margin_class: Optional[str] = None
    storage_recommended: Optional[bool] = None


# =============================================================================
# PRIMARY RESPONSE SCHEMAS
# =============================================================================

class AdvisoryResponse(BaseModel):
    """
    Full response for POST /api/v1/advisory and GET /api/v1/advisory/{id}.
    Contains the complete advisory plus the full reasoning trace.
    """

    # Session identifiers
    session_id: str
    advisory_id: Optional[str] = None

    # Status
    status: str         # completed | pending | failed

    # Core advisory output — what the farmer sees
    recommendation: str
    confidence: float
    reasoning: str
    sowing_advice: str
    irrigation_advice: str
    market_advice: str

    # Structured outputs
    risk_flags: list[RiskFlag] = []
    actions: list[ActionItem] = []

    # Reasoning trace — the "show your work" layer for the demo
    reasoning_trace: list[ReasoningStepResponse] = []

    # Tool summaries — compact data snapshots for the UI
    weather_summary: Optional[dict] = None
    crop_summary: Optional[dict] = None
    soil_summary: Optional[dict] = None
    market_summary: Optional[dict] = None

    # Metadata
    total_duration_ms: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        # Allow datetime objects to be serialised to ISO strings
        json_encoders = {datetime: lambda v: v.isoformat()}


class AdvisoryListItem(BaseModel):
    """
    Compact advisory item for GET /api/v1/sessions list view.
    Used in the demo history sidebar — not the full detail view.
    """
    session_id: str
    crop: str
    location_name: str
    status: str
    recommendation: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CropListItem(BaseModel):
    """One crop in the supported crops list for the form dropdown."""
    id: str             # e.g. "wheat"
    name: str           # e.g. "Wheat"
    local_names: list[str] = []
    category: str       # kharif | rabi | vegetable_annual
    regions_available: list[str] = []


class RegionListItem(BaseModel):
    """One region in the supported regions list for the form dropdown."""
    id: str             # e.g. "north_india"
    name: str           # e.g. "North India (Punjab, Haryana, UP, Uttarakhand)"


class SoilTypeListItem(BaseModel):
    """One soil type for the form dropdown."""
    id: str             # e.g. "black_cotton"
    name: str           # e.g. "Black Cotton Soil (Regur / Vertisol)"
    local_names: list[str] = []
    drainage: str
    waterlogging_risk: str
    irrigation_multiplier: float


# =============================================================================
# ERROR AND HEALTH SCHEMAS
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response shape for all 4xx and 5xx responses."""
    error: str          # machine-readable error code e.g. "validation_error"
    message: str        # human-readable description
    details: Optional[dict] = None  # optional extra context


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health — used by Azure liveness probe."""
    status: str         # "healthy" | "degraded" | "unhealthy"
    version: str        # app version from config
    database: str       # "connected" | "disconnected"
    llm_endpoint: str   # GitHub Models base URL for confirmation
    environment: str    # "development" | "production"