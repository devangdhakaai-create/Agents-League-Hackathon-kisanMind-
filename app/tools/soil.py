# =============================================================================
# app/tools/soil.py
# =============================================================================
#
# PURPOSE:
#   Loads the soil knowledge base (soil_profiles.json) and provides
#   structured soil data to the reasoning engine on demand.
#
# ARCHITECTURAL ROLE:
#   Static data tool — no external API calls, no database reads.
#   soil_profiles.json is loaded ONCE at module import into the module-level
#   _SOIL_DATA variable. Every call to get_soil_profile() reads from memory.
#
# KEY RESPONSIBILITIES:
#   1. Validate requested soil type exists in the knowledge base
#   2. Return the full soil profile shaped for the reasoning engine
#   3. Compute soil-adjusted irrigation interval when base interval is given
#   4. Assess crop-soil compatibility and flag mismatches
#   5. Surface waterlogging and drainage warnings proactively
#
# DESIGN DECISION — Why compute irrigation interval here?
#   The reasoning engine could multiply: base_interval × multiplier.
#   But that requires the LLM to retrieve two values from two tool results
#   and compute the product. Doing it here means the tool result already
#   contains the final answer: "irrigate every 29 days."
#   The LLM cites this number directly in the advisory. No arithmetic needed.
#
# =============================================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING — MODULE LEVEL
# =============================================================================

_DATA_PATH = Path(__file__).parent.parent / "data" / "soil_profiles.json"

try:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        _SOIL_DATA: dict = json.load(f)

    _SOIL_PROFILES: dict = _SOIL_DATA.get("soil_profiles", {})
    _MULTIPLIER_INDEX: dict = _SOIL_DATA.get("irrigation_multiplier_summary", {})
    _WATERLOGGING_INDEX: dict = _SOIL_DATA.get("waterlogging_risk_summary", {})

    logger.info(
        f"Soil knowledge base loaded: {len(_SOIL_PROFILES)} profiles "
        f"({', '.join(_SOIL_PROFILES.keys())})"
    )

except FileNotFoundError:
    logger.critical(f"soil_profiles.json not found at {_DATA_PATH}. Cannot start.")
    raise
except json.JSONDecodeError as e:
    logger.critical(f"soil_profiles.json is malformed: {e}. Cannot start.")
    raise


# =============================================================================
# SUPPORTED SOIL TYPES
# =============================================================================

SUPPORTED_SOIL_TYPES = list(_SOIL_PROFILES.keys())

# Display names for the frontend form dropdown
SOIL_TYPE_DISPLAY_NAMES = {
    "black_cotton": "Black Cotton Soil (Regur / Vertisol)",
    "loamy":        "Loamy Soil (Alluvial Loam)",
    "sandy":        "Sandy Soil (Arid / Desert Soil)",
    "clay":         "Clay Soil (Heavy Alluvial Clay)",
    "red_laterite": "Red Laterite Soil (Laterite / Ultisol)",
    "sandy_loam":   "Sandy Loam Soil (Light Alluvial)",
}

# Waterlogging risk severity ordering — used for comparison and flagging
_WATERLOGGING_SEVERITY = {
    "very_low": 0,
    "low":      1,
    "medium":   2,
    "high":     3,
    "very_high": 4,
}


# =============================================================================
# MAIN TOOL FUNCTION
# =============================================================================

def get_soil_profile(
    soil_type: str,
    base_irrigation_interval_days: Optional[int] = None,
    crop: Optional[str] = None,
) -> dict:
    """
    Returns structured soil profile data for the reasoning engine.

    ARGUMENTS:
        soil_type                     → soil identifier, must match a key
                                        in soil_profiles.json
                                        e.g. "black_cotton", "loamy", "sandy"

        base_irrigation_interval_days → optional: the crop's base irrigation
                                        interval from get_crop_data().
                                        If provided, the soil-adjusted interval
                                        is computed and returned as a ready-to-use
                                        recommendation.

        crop                          → optional: the crop being grown.
                                        If provided, a crop-soil compatibility
                                        assessment is included in the response.

    RETURNS:
        dict shaped for the reasoning engine. Never raises — returns error
        dict on invalid input.

    NORMALISATION:
        soil_type is lowercased and stripped. Spaces replaced with underscores.
        "Black Cotton", "BLACK_COTTON", "black cotton" all resolve correctly.
    """
    soil_type = soil_type.lower().strip().replace(" ", "_").replace("-", "_")

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    if soil_type not in _SOIL_PROFILES:
        return _build_error_response(
            soil_type=soil_type,
            error=f"Soil type '{soil_type}' not found in knowledge base.",
            suggestion=(
                f"Supported types: {', '.join(sorted(SUPPORTED_SOIL_TYPES))}. "
                f"Use underscore format: 'black_cotton', 'sandy_loam', etc."
            ),
        )

    profile = _SOIL_PROFILES[soil_type]

    # ------------------------------------------------------------------
    # PHYSICAL PROPERTIES
    # ------------------------------------------------------------------

    physical = profile.get("physical_properties", {})
    agronomic = profile.get("agronomic_properties", {})
    irrigation_guidance = profile.get("irrigation_guidance", {})

    # ------------------------------------------------------------------
    # IRRIGATION MULTIPLIER AND ADJUSTED INTERVAL
    # ------------------------------------------------------------------

    multiplier = _MULTIPLIER_INDEX.get(soil_type, 1.0)
    irrigation_section = _build_irrigation_section(
        soil_type=soil_type,
        multiplier=multiplier,
        irrigation_guidance=irrigation_guidance,
        base_interval=base_irrigation_interval_days,
    )

    # ------------------------------------------------------------------
    # WATERLOGGING RISK ASSESSMENT
    # ------------------------------------------------------------------

    waterlogging_risk = _WATERLOGGING_INDEX.get(soil_type, "unknown")
    waterlogging_section = _build_waterlogging_section(
        soil_type=soil_type,
        waterlogging_risk=waterlogging_risk,
        irrigation_guidance=irrigation_guidance,
    )

    # ------------------------------------------------------------------
    # CROP COMPATIBILITY (optional — only if crop is provided)
    # ------------------------------------------------------------------

    compatibility_section = None
    if crop:
        compatibility_section = _assess_crop_soil_compatibility(
            crop=crop,
            soil_type=soil_type,
            profile=profile,
        )

    # ------------------------------------------------------------------
    # WATER STRESS SIGNALS
    # ------------------------------------------------------------------

    stress_signals = profile.get("water_stress_signals", [])

    # ------------------------------------------------------------------
    # ASSEMBLE RESPONSE
    # ------------------------------------------------------------------

    response = {
        "tool": "get_soil_profile",
        "status": "success",
        "soil_type": soil_type,
        "soil_display_name": SOIL_TYPE_DISPLAY_NAMES.get(
            soil_type, profile.get("name", soil_type.replace("_", " ").title())
        ),
        "local_names": profile.get("local_names", []),
        "distribution": profile.get("distribution", ""),

        # Physical properties — what the soil is made of
        "physical_properties": {
            "texture": physical.get("texture"),
            "drainage": physical.get("drainage"),
            "field_capacity_percent": physical.get("field_capacity_percent"),
            "wilting_point_percent": physical.get("wilting_point_percent"),
            "available_water_capacity_percent": physical.get(
                "available_water_capacity_percent"
            ),
            "infiltration_rate_mm_per_hour": physical.get(
                "infiltration_rate_mm_per_hour"
            ),
            "depth_cm": physical.get("depth_cm"),
            "notes": physical.get("notes", ""),
        },

        # Agronomic properties — fertility and pH
        "agronomic_properties": {
            "ph_range": f"{agronomic.get('ph_range_min', 'N/A')}–{agronomic.get('ph_range_max', 'N/A')}",
            "ph_min": agronomic.get("ph_range_min"),
            "ph_max": agronomic.get("ph_range_max"),
            "fertility": agronomic.get("fertility"),
            "nitrogen_level": agronomic.get("nitrogen"),
            "phosphorus_level": agronomic.get("phosphorus"),
            "potassium_level": agronomic.get("potassium"),
            "organic_matter": agronomic.get("organic_matter"),
            "notes": agronomic.get("notes", ""),
        },

        # Irrigation — the most important section for advisory output
        "irrigation": irrigation_section,

        # Waterlogging — critical risk factor for several soil types
        "waterlogging": waterlogging_section,

        # Crop compatibility — only included when crop is provided
        "crop_compatibility": compatibility_section,

        # Water stress signals — observable field indicators for farmers
        "water_stress_signals": stress_signals,

        # Suitable and unsuitable crops from the knowledge base
        "suitable_crops": profile.get("suitable_crops", []),
        "unsuitable_crops": profile.get("unsuitable_crops", []),
        "best_crop_fit": profile.get("best_crop_fit"),

        # Notes for the reasoning engine — direct guidance for LLM
        "notes_for_reasoning_engine": profile.get(
            "notes_for_reasoning_engine", ""
        ),

        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    return response


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_supported_soil_types() -> list[dict]:
    """
    Returns all supported soil types for the frontend form dropdown.
    CALLED BY: routes.py — implicitly via the /crops and /regions endpoints,
    or a dedicated /soils endpoint if you add one.
    """
    return [
        {
            "id": soil_id,
            "name": SOIL_TYPE_DISPLAY_NAMES.get(soil_id, soil_id.replace("_", " ").title()),
            "local_names": _SOIL_PROFILES[soil_id].get("local_names", []),
            "drainage": _SOIL_PROFILES[soil_id]
                        .get("physical_properties", {})
                        .get("drainage", "unknown"),
            "waterlogging_risk": _WATERLOGGING_INDEX.get(soil_id, "unknown"),
            "irrigation_multiplier": _MULTIPLIER_INDEX.get(soil_id, 1.0),
        }
        for soil_id in SUPPORTED_SOIL_TYPES
    ]


def get_irrigation_multiplier(soil_type: str) -> Optional[float]:
    """
    Returns just the irrigation multiplier for a soil type.
    Convenience function used by the agent engine for quick lookups
    without loading the full profile.
    """
    soil_type = soil_type.lower().strip().replace(" ", "_")
    return _MULTIPLIER_INDEX.get(soil_type)


def get_waterlogging_risk(soil_type: str) -> Optional[str]:
    """
    Returns just the waterlogging risk level for a soil type.
    Convenience function for risk flag generation in the agent engine.
    """
    soil_type = soil_type.lower().strip().replace(" ", "_")
    return _WATERLOGGING_INDEX.get(soil_type)


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================

def _build_irrigation_section(
    soil_type: str,
    multiplier: float,
    irrigation_guidance: dict,
    base_interval: Optional[int],
) -> dict:
    """
    Builds the irrigation section of the soil profile response.

    If base_irrigation_interval_days is provided (from get_crop_data()),
    computes the soil-adjusted interval and a human-readable recommendation.

    This is the most important computation this file performs.
    The adjusted interval is the single most actionable number in the
    entire advisory output — "irrigate every N days" is something
    a farmer can act on directly.

    COMPUTATION:
        adjusted_interval = round(base_interval * multiplier)

    EXAMPLE:
        Wheat base interval: 21 days
        Sandy soil multiplier: 0.55
        Adjusted interval: round(21 * 0.55) = 12 days
        Output: "Irrigate wheat every 12 days on sandy soil"

    ROUNDING:
        Rounded to nearest integer because irrigation schedules are
        planned in whole days. A farmer cannot irrigate every 11.55 days.
    """
    section = {
        "irrigation_multiplier": multiplier,
        "multiplier_explanation": irrigation_guidance.get("reasoning", ""),
        "waterlogging_risk_level": irrigation_guidance.get("waterlogging_risk"),
        "preferred_method": irrigation_guidance.get("preferred_irrigation_method"),
        "method_to_avoid": irrigation_guidance.get("avoid_irrigation_method"),
    }

    if base_interval is not None and base_interval > 0:
        adjusted = round(base_interval * multiplier)

        # Ensure adjusted interval is at least 1 day
        adjusted = max(1, adjusted)

        # Determine whether adjustment is significant
        delta = adjusted - base_interval
        if delta < 0:
            direction = f"{abs(delta)} days MORE frequently than the base recommendation"
            context = (
                "This soil drains or evaporates moisture faster than the reference loamy soil. "
                "More frequent irrigation prevents crop stress."
            )
        elif delta > 0:
            direction = f"{delta} days LESS frequently than the base recommendation"
            context = (
                "This soil retains moisture longer than the reference loamy soil. "
                "Less frequent irrigation prevents waterlogging."
            )
        else:
            direction = "at the same frequency as the base recommendation"
            context = (
                "Loamy soil is the reference — no adjustment needed."
            )

        section["base_interval_days"] = base_interval
        section["adjusted_interval_days"] = adjusted
        section["adjustment_delta_days"] = delta
        section["adjustment_direction"] = direction
        section["adjustment_context"] = context
        section["irrigation_recommendation"] = (
            f"Irrigate every {adjusted} days on {soil_type.replace('_', ' ')} soil "
            f"({direction} — base: {base_interval} days)."
        )

    else:
        section["base_interval_days"] = None
        section["adjusted_interval_days"] = None
        section["irrigation_recommendation"] = (
            f"Apply the crop's base irrigation interval × {multiplier} "
            f"to get the soil-adjusted schedule. "
            f"(e.g. if crop base is 21 days → irrigate every "
            f"{round(21 * multiplier)} days on this soil)"
        )

    return section


def _build_waterlogging_section(
    soil_type: str,
    waterlogging_risk: str,
    irrigation_guidance: dict,
) -> dict:
    """
    Builds the waterlogging risk section.

    Waterlogging is one of the highest-severity risks in Indian agriculture.
    Clay and black cotton soils have very high waterlogging risk during
    monsoon — this section surfaces that clearly so the reasoning engine
    always includes waterlogging warnings when relevant.

    SEVERITY LEVELS:
        very_high → always include drainage warning in risk_flags
        high      → include drainage warning if kharif season or heavy rain
        medium    → mention drainage as a precaution
        low/very_low → no drainage warning needed
    """
    severity_score = _WATERLOGGING_SEVERITY.get(waterlogging_risk, 0)

    # Build proactive warnings based on severity
    if severity_score >= 4:  # very_high
        warning = (
            f"{soil_type.replace('_', ' ').title()} has VERY HIGH waterlogging risk. "
            f"Field drainage channels must be maintained before kharif sowing. "
            f"Standing water for 48+ hours causes root damage in most crops. "
            f"Include drainage maintenance as Priority 1 action item."
        )
        include_in_risk_flags = True
        drainage_action_required = True

    elif severity_score >= 3:  # high
        warning = (
            f"{soil_type.replace('_', ' ').title()} has HIGH waterlogging risk. "
            f"Inspect and clear drainage channels before heavy rainfall. "
            f"Monitor soil after rainfall events exceeding 30mm in 24 hours."
        )
        include_in_risk_flags = True
        drainage_action_required = True

    elif severity_score >= 2:  # medium
        warning = (
            f"{soil_type.replace('_', ' ').title()} has moderate waterlogging risk. "
            f"Ensure adequate field drainage for kharif crops. "
            f"Flat fields with no drainage slope require bunding."
        )
        include_in_risk_flags = False
        drainage_action_required = False

    else:  # low / very_low
        warning = (
            f"{soil_type.replace('_', ' ').title()} has low waterlogging risk. "
            f"Good natural drainage — waterlogging unlikely under normal rainfall."
        )
        include_in_risk_flags = False
        drainage_action_required = False

    return {
        "risk_level": waterlogging_risk,
        "severity_score": severity_score,
        "warning": warning,
        "include_in_risk_flags": include_in_risk_flags,
        "drainage_action_required": drainage_action_required,
        "waterlogging_note": irrigation_guidance.get("waterlogging_note", ""),
    }


def _assess_crop_soil_compatibility(
    crop: str,
    soil_type: str,
    profile: dict,
) -> dict:
    """
    Assesses whether a specific crop is compatible with this soil type.

    Uses the soil profile's suitable/unsuitable crop lists to determine
    compatibility level. Returns a structured assessment the reasoning
    engine can include in its reasoning chain.

    COMPATIBILITY LEVELS:
        excellent  → soil is the best fit for this crop
        good       → crop is in the suitable list
        marginal   → crop not listed as suitable but not listed as unsuitable
        poor       → crop is in the unsuitable list (include strong warning)

    NOTE:
        This is a coarse check based on the soil profile's crop lists.
        The fine-grained crop-soil compatibility check using the crop's
        own soil preferences is in crop.py → get_crop_soil_fit().
        Both should agree — if they disagree, the crop.py check takes
        precedence as it is more specific.
    """
    crop_normalised = crop.lower().strip()
    best_fit = profile.get("best_crop_fit", "").lower()
    suitable = [c.lower() for c in profile.get("suitable_crops", [])]
    unsuitable = [c.lower() for c in profile.get("unsuitable_crops", [])]

    # Check best fit
    if crop_normalised == best_fit:
        return {
            "compatibility": "excellent",
            "message": (
                f"{crop.title()} is the BEST FIT crop for "
                f"{soil_type.replace('_', ' ')} soil. "
                f"Optimal agronomic conditions — no soil management concerns."
            ),
            "flag_in_advisory": False,
            "recommended_actions": [],
        }

    # Check suitable list
    if any(crop_normalised in s for s in suitable):
        return {
            "compatibility": "good",
            "message": (
                f"{crop.title()} is well-suited to "
                f"{soil_type.replace('_', ' ')} soil. "
                f"Standard management practices apply."
            ),
            "flag_in_advisory": False,
            "recommended_actions": [],
        }

    # Check unsuitable list
    if any(crop_normalised in u for u in unsuitable):
        actions = _get_mitigation_actions(crop_normalised, soil_type)
        return {
            "compatibility": "poor",
            "message": (
                f"WARNING: {crop.title()} is NOT well-suited to "
                f"{soil_type.replace('_', ' ')} soil. "
                f"Risk of poor establishment, waterlogging, or crop failure. "
                f"Consider switching to: {profile.get('best_crop_fit', 'a more suitable crop')}."
            ),
            "flag_in_advisory": True,
            "recommended_actions": actions,
        }

    # Not in either list — marginal / insufficient data
    return {
        "compatibility": "marginal",
        "message": (
            f"No specific compatibility data for {crop.title()} on "
            f"{soil_type.replace('_', ' ')} soil. "
            f"Best fit crop for this soil: {profile.get('best_crop_fit', 'unknown')}. "
            f"Proceed with standard management and monitor closely."
        ),
        "flag_in_advisory": False,
        "recommended_actions": [
            "Conduct soil test before sowing to verify pH and nutrient levels.",
            "Start with a small trial plot if this crop-soil combination is new.",
        ],
    }


def _get_mitigation_actions(crop: str, soil_type: str) -> list[str]:
    """
    Returns soil management actions that can partially mitigate a poor
    crop-soil fit. Used when compatibility is "poor" to give the farmer
    constructive options rather than just a warning.

    Not all poor combinations can be mitigated — some require crop switching.
    """
    actions = []

    # Sandy soil mitigations
    if soil_type == "sandy":
        actions.extend([
            "Add 5–8 tonnes/hectare of organic compost before sowing to improve water retention.",
            "Use drip irrigation — reduces water loss through percolation significantly.",
            "Apply mulching to reduce surface evaporation.",
            "Consider raised bed cultivation to concentrate soil improvement effort.",
        ])

    # Clay soil mitigations
    elif soil_type == "clay":
        actions.extend([
            "Construct raised beds or ridges to improve drainage before sowing.",
            "Install subsurface drainage if field is flat and rain-fed.",
            "Avoid tilling when soil is wet — causes compaction.",
            "Add organic matter to improve soil structure over time.",
        ])

    # Black cotton mitigations
    elif soil_type == "black_cotton":
        actions.extend([
            "Ensure field drainage channels are clear before sowing.",
            "Use ridge and furrow planting method to manage waterlogging risk.",
            "Avoid first irrigation until cracks close naturally after first rain.",
        ])

    # Red laterite mitigations
    elif soil_type == "red_laterite":
        actions.extend([
            "Apply lime at 1–2 tonnes/hectare to raise pH above 6.0 for sensitive crops.",
            "Add compost or green manure to improve organic matter and nutrient retention.",
            "Use drip irrigation to manage water efficiently on shallow soils.",
        ])

    # Fallback
    if not actions:
        actions.append(
            "Conduct a soil test and consult local agricultural extension for specific management advice."
        )

    return actions


def _build_error_response(
    soil_type: str,
    error: str,
    suggestion: str = "",
) -> dict:
    """
    Returns a structured error dict when soil data cannot be retrieved.
    Consistent shape with other tool error responses — `status` field
    always present for the reasoning engine to check.
    """
    return {
        "tool": "get_soil_profile",
        "status": "error",
        "soil_type": soil_type,
        "error": error,
        "suggestion": suggestion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }