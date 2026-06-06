# =============================================================================
# app/tools/crop.py
# =============================================================================
#
# PURPOSE:
#   Loads the crop knowledge base (crops.json) and provides structured
#   crop data to the reasoning engine on demand.
#
# ARCHITECTURAL ROLE:
#   Static data tool — no external API calls, no database reads.
#   crops.json is loaded ONCE at module import into the module-level
#   _CROPS_DATA variable. Every subsequent call to get_crop_data() reads
#   from memory — sub-millisecond response time.
#
# KEY RESPONSIBILITIES:
#   1. Validate requested crop exists in the knowledge base
#   2. Select the correct region-specific sowing window
#   3. Compute sowing window status relative to today's date
#   4. Compute days into the growing season if already sown
#   5. Return a clean, reasoning-ready dict
#
# DESIGN DECISION — Why compute sowing window status here?
#   The LLM could compute "is today inside the sowing window?" if given
#   the start/end dates and today's date. But that requires calendar math
#   in the prompt, which wastes tokens and introduces error risk. Doing it
#   in Python here is deterministic, free, and keeps the reasoning prompt
#   focused on agronomic judgment — not date arithmetic.
#
# =============================================================================

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING — MODULE LEVEL
# =============================================================================
#
# Load crops.json exactly once when this module is first imported.
# All subsequent calls to any function in this file read from _CROPS_DATA.
#
# WHY MODULE-LEVEL (not inside the function)?
#   Loading JSON from disk on every tool call would add 2–5ms per call.
#   With 6 reasoning steps per session and a demo, that adds up.
#   More importantly: if the file is missing or malformed, you want to
#   discover that at startup (loud failure), not during a live demo call
#   (silent failure mid-reasoning).
#
# PATH RESOLUTION:
#   Path(__file__) is the absolute path of this file (tools/crop.py).
#   .parent is tools/
#   .parent.parent is app/
#   / "data" / "crops.json" builds: app/data/crops.json
#   This works regardless of where you run the application from.
#
# =============================================================================

_DATA_PATH = Path(__file__).parent.parent / "data" / "crops.json"

try:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        _CROPS_DATA: dict = json.load(f)
    _CROP_REGISTRY: dict = _CROPS_DATA.get("crops", {})
    logger.info(
        f"Crop knowledge base loaded: {len(_CROP_REGISTRY)} crops "
        f"({', '.join(_CROP_REGISTRY.keys())})"
    )
except FileNotFoundError:
    logger.critical(f"crops.json not found at {_DATA_PATH}. Cannot start.")
    raise
except json.JSONDecodeError as e:
    logger.critical(f"crops.json is malformed: {e}. Cannot start.")
    raise


# =============================================================================
# SUPPORTED REGIONS
# =============================================================================
#
# Valid region identifiers. Must match keys used in crops.json `regions` dicts.
# The frontend form uses these as dropdown options.
# The reasoning engine passes one of these when calling get_crop_data().
#
# =============================================================================

SUPPORTED_REGIONS = [
    "north_india",
    "south_india",
    "west_india",
    "east_india",
    "central_india",
]

# Region display names for the frontend form
REGION_DISPLAY_NAMES = {
    "north_india":   "North India (Punjab, Haryana, UP, Uttarakhand)",
    "south_india":   "South India (Tamil Nadu, Karnataka, AP, Telangana, Kerala)",
    "west_india":    "West India (Gujarat, Maharashtra, Rajasthan)",
    "east_india":    "East India (West Bengal, Odisha, Bihar, Jharkhand)",
    "central_india": "Central India (Madhya Pradesh, Chhattisgarh, Vidarbha)",
}


# =============================================================================
# MAIN TOOL FUNCTION
# =============================================================================

def get_crop_data(crop: str, region: str) -> dict:
    """
    Returns structured crop advisory data for the reasoning engine.

    The reasoning engine calls this tool once per session to get:
    - Sowing window for the region
    - Sowing window status (in_window / approaching / missed / off_season)
    - Days until window opens or closes
    - Water requirements and critical irrigation stages
    - Soil compatibility
    - Risk factors relevant to current season
    - Market seasonality signals

    ARGUMENTS:
        crop   → crop identifier, must match a key in crops.json
                 e.g. "wheat", "rice", "cotton", "maize", "tomato", "soybean"
        region → region identifier, must be in SUPPORTED_REGIONS
                 e.g. "north_india", "south_india"

    RETURNS:
        dict with full crop advisory data, shaped for the reasoning engine.
        Never raises — returns an error dict on invalid input.

    NORMALISATION:
        Crop and region strings are lowercased and stripped before lookup.
        "Wheat", "WHEAT", " wheat " all resolve to "wheat".
    """
    crop = crop.lower().strip().replace(" ", "_")
    region = region.lower().strip().replace(" ", "_")

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    if crop not in _CROP_REGISTRY:
        return _build_error_response(
            crop=crop,
            region=region,
            error=f"Crop '{crop}' not found in knowledge base.",
            suggestion=(
                f"Supported crops: {', '.join(sorted(_CROP_REGISTRY.keys()))}. "
                f"Use the exact identifier (e.g. 'wheat', 'rice', 'cotton')."
            ),
        )

    if region not in SUPPORTED_REGIONS:
        return _build_error_response(
            crop=crop,
            region=region,
            error=f"Region '{region}' not recognised.",
            suggestion=(
                f"Supported regions: {', '.join(SUPPORTED_REGIONS)}."
            ),
        )

    crop_profile = _CROP_REGISTRY[crop]

    # ------------------------------------------------------------------
    # REGION-SPECIFIC SOWING WINDOW
    # ------------------------------------------------------------------
    # Not all crops have data for all regions.
    # If the exact region is missing, fall back to the closest match.
    # If no match exists at all, return a graceful error.
    # ------------------------------------------------------------------

    region_data, resolved_region = _resolve_region(crop_profile, region, crop)

    if region_data is None:
        return _build_error_response(
            crop=crop,
            region=region,
            error=(
                f"No regional data for '{crop}' in '{region}'. "
                f"Available regions for this crop: "
                f"{', '.join(crop_profile.get('regions', {}).keys())}."
            ),
            suggestion="Try a neighbouring region or check if this crop is grown in your area.",
        )

    # ------------------------------------------------------------------
    # SOWING WINDOW STATUS
    # ------------------------------------------------------------------

    today = date.today()
    sowing_status = _compute_sowing_window_status(
        today=today,
        sow_start_str=region_data.get("sow_window_start"),
        sow_end_str=region_data.get("sow_window_end"),
    )

    # ------------------------------------------------------------------
    # HARVEST WINDOW
    # ------------------------------------------------------------------

    harvest_info = _compute_harvest_info(
        today=today,
        harvest_start_str=region_data.get("harvest_window_start"),
        harvest_end_str=region_data.get("harvest_window_end"),
        growing_days=region_data.get("growing_days"),
    )

    # ------------------------------------------------------------------
    # IRRIGATION COMPUTATION
    # ------------------------------------------------------------------
    # Extract the most critical upcoming irrigation stage based on today's
    # date and the crop's stage calendar (days_after_sowing).
    # This is used by the reasoning engine to prioritise irrigation advice.
    # ------------------------------------------------------------------

    water_data = crop_profile.get("water", {})
    critical_stages = water_data.get("critical_stages", [])

    # ------------------------------------------------------------------
    # SOIL COMPATIBILITY SUMMARY
    # ------------------------------------------------------------------

    soil_data = crop_profile.get("soil", {})
    soil_summary = {
        "preferred_soils": soil_data.get("preferred", []),
        "tolerated_soils": soil_data.get("tolerated", []),
        "avoid_soils": soil_data.get("avoid", []),
        "optimal_ph": f"{soil_data.get('optimal_ph_min', 'N/A')}–{soil_data.get('optimal_ph_max', 'N/A')}",
        "notes": soil_data.get("notes", ""),
    }

    # ------------------------------------------------------------------
    # RISK FACTORS — SEASON FILTERED
    # ------------------------------------------------------------------
    # Return all risk factors but flag which are seasonally relevant.
    # A harvest-rain risk is not relevant during sowing season.
    # The reasoning engine uses seasonally_active to prioritise warnings.
    # ------------------------------------------------------------------

    risk_factors = _annotate_risks(
        risks=crop_profile.get("risk_factors", []),
        sowing_status=sowing_status["status"],
    )

    # ------------------------------------------------------------------
    # MARKET SEASONALITY
    # ------------------------------------------------------------------

    market_data = crop_profile.get("market", {})

    # ------------------------------------------------------------------
    # ASSEMBLE RESPONSE
    # ------------------------------------------------------------------

    return {
        "tool": "get_crop_data",
        "status": "success",
        "crop": crop,
        "crop_display_name": crop_profile.get("name", crop.title()),
        "local_names": crop_profile.get("local_names", []),
        "category": crop_profile.get("category", "unknown"),
        "description": crop_profile.get("description", ""),

        # Regional sowing window
        "region": region,
        "resolved_region": resolved_region,
        "region_notes": region_data.get("notes", ""),
        "growing_days": region_data.get("growing_days"),

        # Sowing window status — the most important field for the agent
        "sowing_window": {
            "start": region_data.get("sow_window_start"),
            "end": region_data.get("sow_window_end"),
            "status": sowing_status["status"],
            "days_until_open": sowing_status.get("days_until_open"),
            "days_until_close": sowing_status.get("days_until_close"),
            "days_past_close": sowing_status.get("days_past_close"),
            "urgency_message": sowing_status.get("urgency_message"),
            "recommendation": sowing_status.get("recommendation"),
        },

        # Harvest window
        "harvest_window": harvest_info,

        # Water requirements
        "water_requirements": {
            "total_water_need_mm": water_data.get("total_water_need_mm"),
            "base_irrigation_interval_days": water_data.get("irrigation_interval_days"),
            "drought_tolerance": water_data.get("drought_tolerance"),
            "waterlogging_tolerance": water_data.get("waterlogging_tolerance"),
            "critical_stages": critical_stages,
            "note": (
                "Multiply base_irrigation_interval_days by the soil's "
                "irrigation_multiplier (from soil tool) to get the "
                "soil-adjusted irrigation interval."
            ),
        },

        # Soil compatibility
        "soil_compatibility": soil_summary,

        # Risk factors with seasonal annotation
        "risk_factors": risk_factors,

        # Market seasonality
        "market_seasonality": {
            "msp_crop": market_data.get("msp_crop"),
            "peak_price_months": market_data.get("peak_price_months", []),
            "off_peak_months": market_data.get("off_peak_months", []),
            "price_volatility": market_data.get("price_volatility"),
            "notes": market_data.get("notes", ""),
        },

        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# API UTILITY FUNCTIONS
# =============================================================================

def list_supported_crops() -> list[dict]:
    """
    Returns a list of all supported crops for the frontend form dropdown.

    CALLED BY: routes.py — GET /api/v1/crops
    Returns display name, identifier, category, and local names.
    """
    crops = []
    for crop_id, profile in _CROP_REGISTRY.items():
        crops.append({
            "id": crop_id,
            "name": profile.get("name", crop_id.title()),
            "local_names": profile.get("local_names", []),
            "category": profile.get("category", "unknown"),
            "regions_available": list(profile.get("regions", {}).keys()),
        })
    return sorted(crops, key=lambda c: c["name"])


def list_supported_regions() -> list[dict]:
    """
    Returns the list of supported regions for the frontend form dropdown.

    CALLED BY: routes.py — GET /api/v1/regions
    """
    return [
        {"id": region_id, "name": REGION_DISPLAY_NAMES[region_id]}
        for region_id in SUPPORTED_REGIONS
    ]


def get_crop_soil_fit(crop: str, soil_type: str) -> dict:
    """
    Returns a compatibility assessment for a crop-soil combination.

    CALLED BY: agent/engine.py (optionally) to pre-validate the combination
    before running the full reasoning loop. Prevents the agent from producing
    an advisory for an agronomically invalid combination without flagging it.

    RETURNS:
        {
          "fit": "preferred" | "tolerated" | "avoid" | "unknown",
          "message": str
        }
    """
    crop = crop.lower().strip()
    soil_type = soil_type.lower().strip()

    if crop not in _CROP_REGISTRY:
        return {"fit": "unknown", "message": f"Crop '{crop}' not in knowledge base."}

    soil_data = _CROP_REGISTRY[crop].get("soil", {})
    preferred = [s.lower() for s in soil_data.get("preferred", [])]
    tolerated = [s.lower() for s in soil_data.get("tolerated", [])]
    avoid = [s.lower() for s in soil_data.get("avoid", [])]

    if soil_type in preferred:
        return {
            "fit": "preferred",
            "message": f"{crop.title()} grows best on {soil_type} soil. Excellent combination.",
        }
    elif soil_type in tolerated:
        return {
            "fit": "tolerated",
            "message": (
                f"{crop.title()} can grow on {soil_type} soil with proper management. "
                f"Preferred soils: {', '.join(preferred)}."
            ),
        }
    elif any(soil_type in a for a in avoid):
        return {
            "fit": "avoid",
            "message": (
                f"{soil_type} soil is not recommended for {crop.title()}. "
                f"Risk of poor establishment or crop failure. "
                f"Preferred soils: {', '.join(preferred)}."
            ),
        }
    else:
        return {
            "fit": "unknown",
            "message": (
                f"No specific compatibility data for {crop.title()} on {soil_type} soil. "
                f"Preferred soils: {', '.join(preferred)}."
            ),
        }


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================

def _resolve_region(
    crop_profile: dict,
    requested_region: str,
    crop: str,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Finds regional data for a crop, falling back to the nearest region
    if the exact requested region has no data.

    FALLBACK LOGIC:
        1. Exact match: north_india → north_india data
        2. Partial match: east_india → north_india (neighbouring)
        3. Any available region → use first available with a warning

    Returns (region_data_dict, resolved_region_name) or (None, None).
    """
    regions = crop_profile.get("regions", {})

    # Exact match
    if requested_region in regions:
        return regions[requested_region], requested_region

    # Fallback map — closest agricultural equivalents
    fallback_map = {
        "east_india":    ["north_india", "central_india"],
        "north_india":   ["central_india", "west_india"],
        "west_india":    ["central_india", "north_india"],
        "south_india":   ["central_india", "west_india"],
        "central_india": ["north_india", "west_india"],
    }

    for fallback in fallback_map.get(requested_region, []):
        if fallback in regions:
            logger.warning(
                f"No data for {crop}/{requested_region}. "
                f"Falling back to {fallback}."
            )
            return regions[fallback], fallback

    # Last resort: use any available region
    if regions:
        fallback_region = list(regions.keys())[0]
        logger.warning(
            f"No close fallback for {crop}/{requested_region}. "
            f"Using {fallback_region} as best available."
        )
        return regions[fallback_region], fallback_region

    return None, None


def _compute_sowing_window_status(
    today: date,
    sow_start_str: Optional[str],
    sow_end_str: Optional[str],
) -> dict:
    """
    Computes whether today falls inside, before, or after the sowing window.

    DATE FORMAT IN crops.json: "MM-DD" (e.g. "11-01" = November 1st)
    We reconstruct with the current year, handling year-wrap for crops
    whose window spans December–January.

    SOWING STATUS VALUES:
        in_window    → sow now, optimal conditions
        approaching  → window opens within 14 days, prepare
        just_missed  → window closed within 14 days, late sowing still possible
        off_season   → window far away, wrong season entirely
        data_missing → date strings not available

    WHY 14-DAY THRESHOLDS?
    Agricultural decisions are made in 1–2 week horizons. A window opening
    in 10 days means "prepare now." A window that closed 10 days ago means
    "late sowing with yield penalty." Both are actionable. Beyond 14 days,
    the information becomes planning-horizon rather than action-horizon.
    """
    if not sow_start_str or not sow_end_str:
        return {
            "status": "data_missing",
            "urgency_message": "Sowing window dates not available for this region.",
            "recommendation": "Consult local agricultural extension office for sowing timing.",
        }

    try:
        year = today.year
        start_month, start_day = map(int, sow_start_str.split("-"))
        end_month, end_day = map(int, sow_end_str.split("-"))

        sow_start = date(year, start_month, start_day)
        sow_end = date(year, end_month, end_day)

        # Handle year wrap: if the window end is before start
        # (e.g. sow_start = Nov 15, sow_end = Feb 28 → spans new year)
        if sow_end < sow_start:
            # If we're past November, the end date is next year
            if today.month >= start_month:
                sow_end = date(year + 1, end_month, end_day)
            else:
                # We're in the early months — start date was last year
                sow_start = date(year - 1, start_month, start_day)

    except (ValueError, AttributeError) as e:
        logger.warning(f"Could not parse sowing dates '{sow_start_str}'–'{sow_end_str}': {e}")
        return {
            "status": "data_missing",
            "urgency_message": "Could not parse sowing window dates.",
            "recommendation": "Check local crop calendar for accurate sowing timing.",
        }

    days_until_open = (sow_start - today).days
    days_until_close = (sow_end - today).days
    days_past_close = (today - sow_end).days

    if sow_start <= today <= sow_end:
        # Inside window
        if days_until_close <= 7:
            urgency = f"URGENT: Sowing window closes in {days_until_close} days. Sow immediately."
            recommendation = "Sow within the next 2–3 days to capture full yield potential."
        elif days_until_close <= 14:
            urgency = f"Window closing soon: {days_until_close} days remaining. Prioritise sowing."
            recommendation = "Complete sowing preparation this week. Target sowing within 7 days."
        else:
            urgency = f"Optimal sowing period. {days_until_close} days remaining in window."
            recommendation = "Current conditions are within optimal sowing window. Proceed when weather and soil conditions are suitable."

        return {
            "status": "in_window",
            "days_until_open": 0,
            "days_until_close": days_until_close,
            "days_past_close": 0,
            "urgency_message": urgency,
            "recommendation": recommendation,
        }

    elif days_until_open > 0:
        # Before window opens
        if days_until_open <= 14:
            urgency = f"Sowing window opens in {days_until_open} days. Begin preparation now."
            recommendation = "Prepare seedbed, procure seeds and inputs. Sowing begins soon."
            status = "approaching"
        else:
            urgency = f"Sowing window opens in {days_until_open} days. Plan ahead."
            recommendation = "Not yet time to sow. Use this period for soil preparation, input procurement, and market research."
            status = "off_season"

        return {
            "status": status,
            "days_until_open": days_until_open,
            "days_until_close": None,
            "days_past_close": 0,
            "urgency_message": urgency,
            "recommendation": recommendation,
        }

    else:
        # Past window close
        if days_past_close <= 14:
            urgency = f"Sowing window closed {days_past_close} days ago. Late sowing carries yield penalty."
            recommendation = (
                f"Late sowing is still possible but expect 10–20% yield reduction. "
                f"Use short-duration varieties if available. "
                f"Consider whether input costs are justified at reduced yield."
            )
            status = "just_missed"
        else:
            urgency = f"Sowing window passed {days_past_close} days ago. This season is closed."
            recommendation = "This crop season has passed. Plan for next season. Consider an alternative crop suitable for the current date."
            status = "off_season"

        return {
            "status": status,
            "days_until_open": None,
            "days_until_close": None,
            "days_past_close": days_past_close,
            "urgency_message": urgency,
            "recommendation": recommendation,
        }


def _compute_harvest_info(
    today: date,
    harvest_start_str: Optional[str],
    harvest_end_str: Optional[str],
    growing_days: Optional[int],
) -> dict:
    """
    Returns harvest window information and days until harvest.
    Used by the reasoning engine to frame market timing advice:
    "Harvest is 120 days away — current cotton price trend suggests
     holding for the Jan–March peak window."
    """
    if not harvest_start_str or not harvest_end_str:
        return {
            "window_start": None,
            "window_end": None,
            "days_until_harvest_start": None,
            "note": "Harvest window data not available for this region.",
        }

    try:
        year = today.year
        h_month, h_day = map(int, harvest_start_str.split("-"))
        he_month, he_day = map(int, harvest_end_str.split("-"))

        harvest_start = date(year, h_month, h_day)
        harvest_end = date(year, he_month, he_day)

        # If harvest window is in the past this year, project to next year
        if harvest_end < today:
            harvest_start = date(year + 1, h_month, h_day)
            harvest_end = date(year + 1, he_month, he_day)

        days_until = (harvest_start - today).days

        return {
            "window_start": harvest_start_str,
            "window_end": harvest_end_str,
            "projected_harvest_start": harvest_start.isoformat(),
            "days_until_harvest_start": max(0, days_until),
            "growing_days": growing_days,
            "note": (
                f"Harvest begins approximately {max(0, days_until)} days from today "
                f"if sowing occurs now."
                if days_until > 0
                else "Harvest window is currently active or recently passed."
            ),
        }

    except (ValueError, AttributeError):
        return {
            "window_start": harvest_start_str,
            "window_end": harvest_end_str,
            "days_until_harvest_start": None,
            "note": "Could not compute harvest timing from available dates.",
        }


def _annotate_risks(risks: list, sowing_status: str) -> list:
    """
    Annotates each risk factor with whether it is seasonally active.

    Seasonal relevance mapping:
        in_window / approaching → flag sowing-time risks
        off_season / just_missed → flag harvest and storage risks
        All statuses → flag pest and weather risks (always relevant)

    The reasoning engine uses `seasonally_active: true` to decide which
    risks to elevate to the `risk_flags` array in the advisory response.
    """
    sowing_phase = sowing_status in ("in_window", "approaching")

    annotated = []
    for risk in risks:
        risk_type = risk.get("type", "")
        condition = risk.get("condition", "")

        # Determine if this risk is currently seasonally relevant
        if risk_type in ("pest", "weather"):
            # Always active — pests and weather are year-round concerns
            active = True
        elif risk_type == "timing" and "sowing" in condition:
            # Timing risks for sowing are active when in/approaching window
            active = sowing_phase
        elif risk_type == "timing" and "harvest" in condition:
            # Harvest timing risks are active when out of sowing window
            active = not sowing_phase
        elif risk_type == "water":
            # Water stress risks are always relevant
            active = True
        elif risk_type == "market":
            # Market risks are always relevant for decision making
            active = True
        else:
            active = True

        annotated.append({
            **risk,
            "seasonally_active": active,
        })

    return annotated


def _build_error_response(
    crop: str,
    region: str,
    error: str,
    suggestion: str = "",
) -> dict:
    """
    Returns a structured error dict when crop data cannot be retrieved.
    Same pattern as weather.py error response — consistent shape across
    all tool functions so the reasoning engine always checks `status`.
    """
    return {
        "tool": "get_crop_data",
        "status": "error",
        "crop": crop,
        "region": region,
        "error": error,
        "suggestion": suggestion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }