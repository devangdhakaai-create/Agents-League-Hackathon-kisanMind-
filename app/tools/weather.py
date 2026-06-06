# =============================================================================
# app/tools/weather.py
# =============================================================================
#
# PURPOSE:
#   Fetches 7-day weather forecast from Open-Meteo API, processes raw data
#   into agricultural signals, and returns a structured dict for the
#   reasoning engine.
#
# ARCHITECTURAL ROLE:
#   This is a "tool function" — one of four data-fetching functions the
#   reasoning engine can call during its ReAct loop. Tool functions follow
#   a strict contract:
#     INPUT:  typed arguments from the LLM's tool call
#     OUTPUT: typed dict — always returns something, never raises to the engine
#     SIDE EFFECT: writes result to tool cache in PostgreSQL
#
# OPEN-METEO API:
#   - Free, no API key required
#   - Docs: https://open-meteo.com/en/docs
#   - Rate limit: 10,000 calls/day (well above hackathon needs)
#   - Returns hourly or daily forecast data in JSON
#
# WHY NO API KEY?
#   Open-Meteo is a fully open meteorological API funded by research grants.
#   No registration, no key, no rate limit concern for this scale.
#   This is intentional — removes one external dependency that could fail
#   on demo day.
#
# =============================================================================

import httpx
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.config import settings
from app.db.database import get_db_context
from app.db import crud

logger = logging.getLogger(__name__)

# =============================================================================
# OPEN-METEO FIELD DEFINITIONS
# =============================================================================
#
# Open-Meteo returns data for specific "daily variables". We request exactly
# what we need for agricultural reasoning — nothing more. Each variable maps
# to a named field in the API response's `daily` object.
#
# VARIABLES REQUESTED:
#   temperature_2m_max      → daily maximum air temperature (°C)
#   temperature_2m_min      → daily minimum air temperature (°C)
#   precipitation_sum       → total daily rainfall (mm)
#   wind_speed_10m_max      → max wind speed at 10m height (km/h)
#   relative_humidity_2m_max → max daily relative humidity (%)
#   relative_humidity_2m_min → min daily relative humidity (%)
#   et0_fao_evapotranspiration → reference evapotranspiration (mm/day)
#                               This is the agronomic gold standard for
#                               irrigation scheduling. ET0 represents water
#                               lost from a reference grass surface — used
#                               to calculate actual crop water demand.
#
# =============================================================================

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "et0_fao_evapotranspiration",
]

# =============================================================================
# AGRICULTURAL RISK THRESHOLDS
# =============================================================================
#
# These constants define the boundaries used to classify weather conditions
# as risky for crop operations. They are based on standard agronomic ranges.
#
# The reasoning engine uses the classified signals (not raw values) to
# generate risk flags. This pre-processing in the tool layer keeps the
# reasoning prompt clean — the LLM receives "heat_stress_risk: true"
# rather than "temperature_max: 38.4°C — is this risky for crops?"
#
# =============================================================================

TEMP_HEAT_STRESS_THRESHOLD_C = 35.0      # above this: heat stress risk for most crops
TEMP_COLD_STRESS_THRESHOLD_C = 10.0      # below this: cold stress / frost risk
RAINFALL_WATERLOGGING_MM_3DAY = 80.0     # >80mm in 3 days: waterlogging risk
RAINFALL_DROUGHT_MM_7DAY = 5.0           # <5mm in 7 days: drought stress risk
HUMIDITY_DISEASE_THRESHOLD_PERCENT = 80  # above this: fungal disease pressure
WIND_CROP_DAMAGE_THRESHOLD_KMH = 40.0   # above this: lodging/physical damage risk


# =============================================================================
# MAIN TOOL FUNCTION
# =============================================================================

def get_weather(
    lat: float,
    lon: float,
    days: int = 7,
) -> dict:
    """
    Fetches weather forecast and returns agricultural signals for the
    reasoning engine.

    This is the primary interface the tool dispatcher calls. All caching,
    API fetching, and signal processing happens inside this function.
    The caller receives a clean, processed dict — never raw API data.

    ARGUMENTS:
        lat   → latitude of the farm location (-90 to 90)
        lon   → longitude of the farm location (-180 to 180)
        days  → number of forecast days (1–16, default 7)

    RETURNS:
        dict with structure defined in _build_agricultural_signals()

    ERROR HANDLING:
        Never raises an exception to the calling engine.
        On any failure, returns a fallback dict with error context so the
        reasoning engine can acknowledge the data gap and reason accordingly.

    CACHE BEHAVIOUR:
        Checks PostgreSQL cache before API call.
        Cache TTL: 1 hour (3600 seconds).
        Cache key: "get_weather:{lat:.4f}:{lon:.4f}:{days}"
    """
    # Round coordinates to 4 decimal places (~11m precision — sufficient
    # for agricultural purposes and improves cache hit rate for nearby farms)
    lat = round(lat, 4)
    lon = round(lon, 4)

    cache_key = f"get_weather:{lat:.4f}:{lon:.4f}:{days}"

    # ------------------------------------------------------------------
    # STEP 1: Check cache
    # ------------------------------------------------------------------
    try:
        with get_db_context() as db:
            cached = crud.get_cached_tool_result(db, cache_key)
            if cached:
                logger.info(f"Weather cache HIT for ({lat}, {lon})")
                return cached
    except Exception as e:
        # Cache failure is non-fatal — proceed to API call
        logger.warning(f"Cache check failed, proceeding to API: {e}")

    # ------------------------------------------------------------------
    # STEP 2: Fetch from Open-Meteo API
    # ------------------------------------------------------------------
    logger.info(f"Fetching weather from Open-Meteo for ({lat}, {lon}), {days} days")

    try:
        raw_data = _fetch_open_meteo(lat, lon, days)
    except Exception as e:
        logger.error(f"Open-Meteo API call failed: {e}")
        return _build_error_response(
            lat=lat,
            lon=lon,
            error=str(e),
            message="Weather data unavailable. Reasoning will proceed with limited weather context."
        )

    # ------------------------------------------------------------------
    # STEP 3: Process raw API data into agricultural signals
    # ------------------------------------------------------------------
    try:
        result = _build_agricultural_signals(raw_data, lat, lon, days)
    except Exception as e:
        logger.error(f"Weather signal processing failed: {e}")
        return _build_error_response(
            lat=lat,
            lon=lon,
            error=str(e),
            message="Weather data received but could not be processed."
        )

    # ------------------------------------------------------------------
    # STEP 4: Write to cache
    # ------------------------------------------------------------------
    try:
        with get_db_context() as db:
            crud.set_cached_tool_result(
                db,
                tool_name="get_weather",
                cache_key=cache_key,
                result=result,
                ttl_seconds=3600,  # 1 hour TTL for weather data
            )
            db.commit()
    except Exception as e:
        # Cache write failure is non-fatal — return result anyway
        logger.warning(f"Cache write failed (non-fatal): {e}")

    logger.info(
        f"Weather fetched successfully for ({lat}, {lon}): "
        f"avg_temp={result['summary']['avg_temp_c']}°C, "
        f"total_rain={result['summary']['total_rainfall_mm']}mm"
    )

    return result


# =============================================================================
# OPEN-METEO API CALL
# =============================================================================

def _fetch_open_meteo(lat: float, lon: float, days: int) -> dict:
    """
    Makes the HTTP GET request to Open-Meteo and returns the raw JSON.

    Uses httpx (synchronous) with a 15-second timeout.
    httpx is preferred over requests for its cleaner API and better
    timeout handling. Both are synchronous here — the reasoning engine
    is not async, so blocking IO is acceptable.

    RAISES:
        httpx.TimeoutException  → API took too long
        httpx.HTTPStatusError   → API returned 4xx or 5xx
        Exception               → any other network error

    WHY NOT ASYNC?
    The reasoning engine's ReAct loop is synchronous Python. Making this
    async would require making the entire engine async, which adds
    complexity without meaningful performance benefit for a single-user
    hackathon demo. Keep it simple.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata",      # IST — standard for Indian farms
        "forecast_days": days,
        "wind_speed_unit": "kmh",        # kmh more intuitive for farmers
        "precipitation_unit": "mm",
    }

    with httpx.Client(timeout=15.0) as client:
        response = client.get(settings.OPEN_METEO_BASE_URL, params=params)
        response.raise_for_status()  # raises HTTPStatusError for 4xx/5xx
        return response.json()


# =============================================================================
# AGRICULTURAL SIGNAL PROCESSING
# =============================================================================

def _build_agricultural_signals(
    raw: dict,
    lat: float,
    lon: float,
    days: int,
) -> dict:
    """
    Converts raw Open-Meteo JSON into agricultural decision signals.

    This is the most important function in this file. The reasoning engine
    does not receive raw API data — it receives processed, classified,
    agriculturally meaningful signals. This design keeps the reasoning
    prompt concise and the LLM focused on agronomic reasoning rather
    than data interpretation.

    OUTPUT STRUCTURE:
        {
          "tool": "get_weather",
          "status": "success",
          "location": { lat, lon },
          "forecast_days": 7,
          "daily": [...],         ← one entry per forecast day
          "summary": {...},       ← aggregated statistics
          "risk_signals": {...},  ← classified boolean risk flags
          "irrigation_signals": {...}, ← irrigation-specific guidance
          "fetched_at": "..."
        }
    """
    daily_data = raw.get("daily", {})

    dates = daily_data.get("time", [])
    temp_max = daily_data.get("temperature_2m_max", [])
    temp_min = daily_data.get("temperature_2m_min", [])
    rainfall = daily_data.get("precipitation_sum", [])
    wind_max = daily_data.get("wind_speed_10m_max", [])
    humidity_max = daily_data.get("relative_humidity_2m_max", [])
    humidity_min = daily_data.get("relative_humidity_2m_min", [])
    et0 = daily_data.get("et0_fao_evapotranspiration", [])

    # ------------------------------------------------------------------
    # BUILD DAILY RECORDS
    # ------------------------------------------------------------------
    # One dict per forecast day, with human-readable field names.
    # None values handled gracefully — Open-Meteo occasionally returns
    # null for future dates in extreme forecast windows.
    # ------------------------------------------------------------------

    daily_records = []
    for i in range(len(dates)):
        record = {
            "date": dates[i] if i < len(dates) else None,
            "temp_max_c": _safe_round(temp_max, i),
            "temp_min_c": _safe_round(temp_min, i),
            "temp_avg_c": _safe_avg(temp_max, temp_min, i),
            "rainfall_mm": _safe_round(rainfall, i),
            "wind_max_kmh": _safe_round(wind_max, i),
            "humidity_max_percent": _safe_round(humidity_max, i),
            "humidity_min_percent": _safe_round(humidity_min, i),
            "et0_mm": _safe_round(et0, i),
            "farming_note": _classify_day(
                temp_max=_safe_get(temp_max, i),
                temp_min=_safe_get(temp_min, i),
                rainfall=_safe_get(rainfall, i),
                humidity_max=_safe_get(humidity_max, i),
                wind=_safe_get(wind_max, i),
            ),
        }
        daily_records.append(record)

    # ------------------------------------------------------------------
    # COMPUTE SUMMARY STATISTICS
    # ------------------------------------------------------------------

    valid_temp_max = [t for t in temp_max if t is not None]
    valid_temp_min = [t for t in temp_min if t is not None]
    valid_rainfall = [r for r in rainfall if r is not None]
    valid_humidity = [h for h in humidity_max if h is not None]
    valid_et0 = [e for e in et0 if e is not None]

    avg_temp = round(
        (sum(valid_temp_max) / len(valid_temp_max) +
         sum(valid_temp_min) / len(valid_temp_min)) / 2, 1
    ) if valid_temp_max and valid_temp_min else None

    total_rainfall = round(sum(valid_rainfall), 1) if valid_rainfall else 0.0
    max_temp = round(max(valid_temp_max), 1) if valid_temp_max else None
    min_temp = round(min(valid_temp_min), 1) if valid_temp_min else None
    max_humidity = round(max(valid_humidity), 1) if valid_humidity else None
    total_et0 = round(sum(valid_et0), 1) if valid_et0 else None

    # Consecutive dry days (no rain forecast)
    dry_day_streak = _count_leading_dry_days(valid_rainfall)

    # 3-day cumulative rainfall for waterlogging assessment
    rainfall_3day = round(sum(valid_rainfall[:3]), 1) if len(valid_rainfall) >= 3 else total_rainfall

    summary = {
        "avg_temp_c": avg_temp,
        "max_temp_c": max_temp,
        "min_temp_c": min_temp,
        "total_rainfall_mm": total_rainfall,
        "rainfall_3day_mm": rainfall_3day,
        "max_humidity_percent": max_humidity,
        "total_et0_mm": total_et0,
        "dry_day_streak": dry_day_streak,
        "forecast_period_days": len(daily_records),
    }

    # ------------------------------------------------------------------
    # RISK SIGNAL CLASSIFICATION
    # ------------------------------------------------------------------
    # Binary flags derived from thresholds defined at the top of this file.
    # The reasoning engine checks these flags to decide which risk_factors
    # from crops.json to elevate in the advisory.
    # ------------------------------------------------------------------

    risk_signals = {
        "heat_stress_risk": (
            max_temp is not None and max_temp > TEMP_HEAT_STRESS_THRESHOLD_C
        ),
        "cold_stress_risk": (
            min_temp is not None and min_temp < TEMP_COLD_STRESS_THRESHOLD_C
        ),
        "waterlogging_risk": (
            rainfall_3day > RAINFALL_WATERLOGGING_MM_3DAY
        ),
        "drought_stress_risk": (
            total_rainfall < RAINFALL_DROUGHT_MM_7DAY
        ),
        "disease_pressure_risk": (
            max_humidity is not None and
            max_humidity > HUMIDITY_DISEASE_THRESHOLD_PERCENT
        ),
        "wind_damage_risk": (
            any(
                w > WIND_CROP_DAMAGE_THRESHOLD_KMH
                for w in wind_max if w is not None
            )
        ),
        "risk_summary": _summarise_risks(
            max_temp, min_temp, rainfall_3day,
            total_rainfall, max_humidity, wind_max
        ),
    }

    # ------------------------------------------------------------------
    # IRRIGATION SIGNALS
    # ------------------------------------------------------------------
    # Pre-computed irrigation guidance based on ET0 and rainfall balance.
    # ET0 (evapotranspiration) represents crop water demand. When rainfall
    # covers less than 70% of ET0, irrigation is required. This follows
    # the FAO-56 simplified water balance approach.
    # ------------------------------------------------------------------

    irrigation_signals = _compute_irrigation_signals(
        total_rainfall=total_rainfall,
        total_et0=total_et0,
        dry_day_streak=dry_day_streak,
    )

    return {
        "tool": "get_weather",
        "status": "success",
        "location": {"lat": lat, "lon": lon},
        "forecast_days": days,
        "daily": daily_records,
        "summary": summary,
        "risk_signals": risk_signals,
        "irrigation_signals": irrigation_signals,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# IRRIGATION SIGNAL COMPUTATION
# =============================================================================

def _compute_irrigation_signals(
    total_rainfall: float,
    total_et0: Optional[float],
    dry_day_streak: int,
) -> dict:
    """
    Computes irrigation need based on rainfall vs evapotranspiration balance.

    FAO-56 SIMPLIFIED WATER BALANCE:
        If rainfall >= 0.7 * ET0 → no irrigation needed (rain covers demand)
        If rainfall < 0.7 * ET0 → irrigation deficit exists
        Deficit mm = ET0 - rainfall (irrigation quantity to apply)

    The reasoning engine uses these signals to generate specific irrigation
    recommendations in the advisory's action items.
    """
    if total_et0 is None or total_et0 == 0:
        return {
            "irrigation_needed": None,
            "deficit_mm": None,
            "reasoning": "ET0 data unavailable — irrigation need cannot be computed from water balance. Use crop stage and visual soil moisture assessment instead.",
            "urgency": "unknown",
        }

    # Rainfall fraction of ET0
    rainfall_fraction = total_rainfall / total_et0

    # Irrigation deficit in mm (how much water the rain has not provided)
    deficit_mm = max(0.0, round(total_et0 - total_rainfall, 1))

    if rainfall_fraction >= 0.9:
        # Rainfall covers >90% of ET0 — irrigation not needed this cycle
        needed = False
        urgency = "none"
        reasoning = (
            f"7-day rainfall ({total_rainfall}mm) covers "
            f"{int(rainfall_fraction*100)}% of crop water demand "
            f"(ET0={total_et0}mm). No irrigation required this cycle."
        )
    elif rainfall_fraction >= 0.7:
        # Rainfall covers 70–90% of ET0 — marginal, monitor closely
        needed = False
        urgency = "monitor"
        reasoning = (
            f"7-day rainfall ({total_rainfall}mm) covers "
            f"{int(rainfall_fraction*100)}% of ET0 ({total_et0}mm). "
            f"Marginal — monitor crop and soil moisture. "
            f"Irrigate if dry streak extends beyond {dry_day_streak + 2} days."
        )
    elif rainfall_fraction >= 0.4:
        # Rainfall covers 40–70% of ET0 — deficit irrigation needed
        needed = True
        urgency = "moderate"
        reasoning = (
            f"Rainfall deficit of {deficit_mm}mm against ET0 of {total_et0}mm. "
            f"Supplemental irrigation of {deficit_mm}mm recommended. "
            f"Current dry streak: {dry_day_streak} days."
        )
    else:
        # Rainfall covers <40% of ET0 — urgent irrigation required
        needed = True
        urgency = "high"
        reasoning = (
            f"Severe irrigation deficit: {deficit_mm}mm required. "
            f"Rainfall ({total_rainfall}mm) covers only "
            f"{int(rainfall_fraction*100)}% of crop water demand. "
            f"Dry streak of {dry_day_streak} days. Irrigate within 24–48 hours "
            f"if crop is in a critical growth stage."
        )

    return {
        "irrigation_needed": needed,
        "deficit_mm": deficit_mm,
        "rainfall_fraction_of_et0": round(rainfall_fraction, 2),
        "urgency": urgency,
        "reasoning": reasoning,
        "dry_day_streak": dry_day_streak,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_get(lst: list, i: int):
    """Returns lst[i] or None if index out of range."""
    return lst[i] if i < len(lst) else None


def _safe_round(lst: list, i: int, decimals: int = 1) -> Optional[float]:
    """Returns rounded lst[i] or None if index out of range or value is None."""
    val = _safe_get(lst, i)
    return round(val, decimals) if val is not None else None


def _safe_avg(lst_a: list, lst_b: list, i: int) -> Optional[float]:
    """Returns average of lst_a[i] and lst_b[i] or None if either is missing."""
    a = _safe_get(lst_a, i)
    b = _safe_get(lst_b, i)
    if a is not None and b is not None:
        return round((a + b) / 2, 1)
    return None


def _count_leading_dry_days(rainfall: list) -> int:
    """
    Counts consecutive dry days from the start of the forecast.
    A day is 'dry' if rainfall < 1mm (standard agronomic threshold).
    Returns 0 if the first day has meaningful rain.
    """
    count = 0
    for r in rainfall:
        if r is None or r < 1.0:
            count += 1
        else:
            break
    return count


def _classify_day(
    temp_max: Optional[float],
    temp_min: Optional[float],
    rainfall: Optional[float],
    humidity_max: Optional[float],
    wind: Optional[float],
) -> str:
    """
    Returns a human-readable farming note for a single forecast day.
    Used in the daily records to give the reasoning engine per-day context.
    """
    notes = []

    if temp_max is not None and temp_max > TEMP_HEAT_STRESS_THRESHOLD_C:
        notes.append(f"heat stress risk ({temp_max}°C)")
    if temp_min is not None and temp_min < TEMP_COLD_STRESS_THRESHOLD_C:
        notes.append(f"cold stress risk ({temp_min}°C)")
    if rainfall is not None and rainfall > 20:
        notes.append(f"heavy rain ({rainfall}mm)")
    elif rainfall is not None and rainfall > 5:
        notes.append(f"moderate rain ({rainfall}mm)")
    if humidity_max is not None and humidity_max > HUMIDITY_DISEASE_THRESHOLD_PERCENT:
        notes.append(f"disease pressure (humidity {humidity_max}%)")
    if wind is not None and wind > WIND_CROP_DAMAGE_THRESHOLD_KMH:
        notes.append(f"strong winds ({wind}km/h)")

    return "; ".join(notes) if notes else "normal farming conditions"


def _summarise_risks(
    max_temp, min_temp, rainfall_3day,
    total_rainfall, max_humidity, wind_max
) -> str:
    """
    Returns a single sentence risk summary for the reasoning engine.
    This is the top-level signal the LLM reads first before examining
    individual risk flags.
    """
    risks = []
    if max_temp and max_temp > TEMP_HEAT_STRESS_THRESHOLD_C:
        risks.append("heat stress")
    if min_temp and min_temp < TEMP_COLD_STRESS_THRESHOLD_C:
        risks.append("cold stress")
    if rainfall_3day > RAINFALL_WATERLOGGING_MM_3DAY:
        risks.append("waterlogging")
    if total_rainfall < RAINFALL_DROUGHT_MM_7DAY:
        risks.append("drought stress")
    if max_humidity and max_humidity > HUMIDITY_DISEASE_THRESHOLD_PERCENT:
        risks.append("disease pressure")
    if wind_max and any(w > WIND_CROP_DAMAGE_THRESHOLD_KMH for w in wind_max if w):
        risks.append("wind damage")

    if not risks:
        return "No significant weather risks detected in the 7-day forecast."
    return f"Weather risks detected: {', '.join(risks)}. Review daily records and adjust operations accordingly."


def _build_error_response(lat: float, lon: float, error: str, message: str) -> dict:
    """
    Returns a structured error dict when the weather API fails.
    The reasoning engine checks `status` before using any data.
    An error response allows the agent to continue with a caveat,
    rather than crashing the entire reasoning loop.
    """
    return {
        "tool": "get_weather",
        "status": "error",
        "location": {"lat": lat, "lon": lon},
        "error": error,
        "message": message,
        "summary": None,
        "risk_signals": None,
        "irrigation_signals": None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }