# =============================================================================
# app/agent/parser.py
# =============================================================================
#
# PURPOSE:
#   Parses the LLM's text response to extract the final_answer JSON object.
#   Also validates the extracted structure against required fields and
#   applies sensible defaults for any missing optional fields.
#
# ARCHITECTURAL ROLE:
#   This file has one job: take raw LLM text output and return a clean,
#   validated Python dict the engine can write to the database and return
#   to the API layer.
#
#   Because we use OpenAI function calling for tool dispatch (handled in
#   engine.py), the parser only needs to handle ONE case: the final_answer.
#   The LLM produces final_answer as a function call with a JSON argument.
#   This file validates and sanitises that JSON.
#
# WHY VALIDATION MATTERS:
#   The LLM can return:
#     - Valid JSON with all fields present        → pass through
#     - Valid JSON with some fields missing       → fill defaults
#     - Valid JSON with wrong types               → coerce or default
#     - Invalid JSON (truncated, malformed)       → attempt repair or fail gracefully
#     - Nested JSON escaped as a string           → parse inner JSON
#
#   Without this file, a single malformed LLM response crashes the entire
#   advisory pipeline. With it, the engine always gets a usable dict back.
#
# =============================================================================

import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# REQUIRED AND OPTIONAL FIELDS
# =============================================================================
#
# REQUIRED: advisory is invalid without these — return error if missing
#           and cannot be inferred.
# OPTIONAL: advisory is degraded but usable — fill with defaults if missing.
#
# =============================================================================

REQUIRED_FIELDS = [
    "recommendation",
    "reasoning",
]

OPTIONAL_FIELDS_WITH_DEFAULTS = {
    "confidence":        0.6,
    "sowing_advice":     "No specific sowing advice generated.",
    "irrigation_advice": "No specific irrigation advice generated.",
    "market_advice":     "No specific market advice generated.",
    "risk_flags":        [],
    "actions":           [],
}

# Valid severity levels for risk flags
VALID_SEVERITIES = {"critical", "high", "medium", "low"}

# Valid risk types
VALID_RISK_TYPES = {"weather", "soil", "market", "timing", "pest", "water"}

# Confidence must be 0.0–1.0
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

# Maximum lengths to prevent runaway LLM verbosity
MAX_RECOMMENDATION_LENGTH = 300
MAX_REASONING_LENGTH = 2000
MAX_ADVICE_LENGTH = 500
MAX_RISK_FLAGS = 8
MAX_ACTIONS = 6


# =============================================================================
# PRIMARY PARSE FUNCTION
# =============================================================================

def parse_final_answer(raw_input) -> dict:
    """
    Parses and validates the LLM's final_answer output.

    ACCEPTS:
        raw_input → either:
            - dict:  already parsed JSON from OpenAI function call arguments
            - str:   raw JSON string that needs parsing
            - None:  returns a fallback error advisory

    RETURNS:
        Validated dict with all required and optional fields present.
        Never raises — always returns a usable dict.

    CALLED BY:
        engine.py — after the LLM returns a final_answer function call.
        The engine passes tool_call.function.arguments (a JSON string)
        directly to this function.
    """
    if raw_input is None:
        logger.warning("parse_final_answer received None input")
        return _build_fallback_advisory(
            reason="LLM returned no final answer content."
        )

    # ------------------------------------------------------------------
    # STEP 1: Ensure we have a dict
    # ------------------------------------------------------------------

    if isinstance(raw_input, dict):
        data = raw_input

    elif isinstance(raw_input, str):
        data = _parse_json_string(raw_input)
        if data is None:
            return _build_fallback_advisory(
                reason=f"Could not parse LLM output as JSON. Raw: {raw_input[:200]}"
            )
    else:
        logger.warning(f"Unexpected raw_input type: {type(raw_input)}")
        return _build_fallback_advisory(
            reason=f"Unexpected input type: {type(raw_input)}"
        )

    # ------------------------------------------------------------------
    # STEP 2: Check required fields
    # ------------------------------------------------------------------

    missing = [f for f in REQUIRED_FIELDS if not data.get(f)]
    if missing:
        logger.warning(f"Final answer missing required fields: {missing}")
        # Attempt to salvage if at least recommendation exists
        if "recommendation" not in data or not data["recommendation"]:
            return _build_fallback_advisory(
                reason=f"LLM final answer missing required fields: {missing}. "
                       f"Raw keys present: {list(data.keys())}"
            )
        # Fill missing reasoning with a placeholder
        if "reasoning" not in data or not data["reasoning"]:
            data["reasoning"] = (
                "Advisory generated from available agricultural data. "
                "Detailed reasoning was not returned by the model."
            )

    # ------------------------------------------------------------------
    # STEP 3: Apply defaults for missing optional fields
    # ------------------------------------------------------------------

    for field, default in OPTIONAL_FIELDS_WITH_DEFAULTS.items():
        if field not in data or data[field] is None:
            data[field] = default
            logger.debug(f"Applied default for missing field: {field}")

    # ------------------------------------------------------------------
    # STEP 4: Sanitise and coerce individual fields
    # ------------------------------------------------------------------

    data = _sanitise_fields(data)

    # ------------------------------------------------------------------
    # STEP 5: Validate and clean risk_flags
    # ------------------------------------------------------------------

    data["risk_flags"] = _validate_risk_flags(data.get("risk_flags", []))

    # ------------------------------------------------------------------
    # STEP 6: Validate and clean actions
    # ------------------------------------------------------------------

    data["actions"] = _validate_actions(data.get("actions", []))

    logger.info(
        f"Final answer parsed: confidence={data['confidence']:.2f}, "
        f"risk_flags={len(data['risk_flags'])}, "
        f"actions={len(data['actions'])}"
    )

    return data


# =============================================================================
# JSON PARSING WITH REPAIR ATTEMPTS
# =============================================================================

def _parse_json_string(raw: str) -> Optional[dict]:
    """
    Attempts to parse a JSON string with multiple fallback strategies.

    STRATEGY 1: Direct json.loads — handles clean JSON strings.
    STRATEGY 2: Strip markdown fences — LLMs sometimes wrap JSON in ```json...```.
    STRATEGY 3: Extract first {...} block — handles prefixed text like "Here is...{...}".
    STRATEGY 4: Repair truncated JSON — add missing closing braces.

    Returns None if all strategies fail.
    """
    if not raw or not raw.strip():
        return None

    # Strategy 1: Direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code fences
    # LLMs sometimes return: ```json\n{...}\n```
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            logger.debug("JSON parsed after stripping markdown fences")
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 3: Extract first complete {...} block
    # Handles cases where LLM adds preamble text before the JSON
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group())
            if isinstance(result, dict):
                logger.debug("JSON parsed by extracting {...} block")
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 4: Attempt to repair truncated JSON
    # Truncation happens when LLM hits max_tokens mid-JSON
    repaired = _attempt_json_repair(raw)
    if repaired:
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                logger.debug("JSON parsed after truncation repair")
                return result
        except json.JSONDecodeError:
            pass

    logger.error(f"All JSON parse strategies failed for input: {raw[:300]}")
    return None


def _attempt_json_repair(raw: str) -> Optional[str]:
    """
    Attempts to repair truncated JSON by closing open braces and brackets.

    TRUNCATION SCENARIO:
        LLM is mid-way through the actions array when max_tokens is hit:
        {"recommendation": "Sow wheat", "actions": [{"priority": 1, "action": "Prep

    REPAIR:
        Count open vs closed braces and brackets, append the missing closers.
        This produces valid JSON for everything the LLM completed,
        even if the last field was truncated.

    NOT A GENERAL JSON REPAIR TOOL:
        This only handles truncation — missing closers at the end.
        It does not fix malformed JSON in the middle of the string.
    """
    # Find the first opening brace — start of JSON object
    start = raw.find("{")
    if start == -1:
        return None

    raw = raw[start:]

    # Count unmatched braces and brackets
    open_braces = raw.count("{") - raw.count("}")
    open_brackets = raw.count("[") - raw.count("]")

    # If JSON appears balanced, return as-is
    if open_braces <= 0 and open_brackets <= 0:
        return raw

    # If the string ends mid-string, close it
    # Check if we're inside a string value (odd number of unescaped quotes)
    repaired = raw.rstrip()

    # Strip trailing comma (common at truncation point)
    if repaired.endswith(","):
        repaired = repaired[:-1]

    # Close open string if truncated mid-value
    # Simple heuristic: if the last non-whitespace char is not a
    # closing bracket/brace/quote, we may be mid-string
    last_char = repaired[-1] if repaired else ""
    if last_char not in ('"}', "}", "]", '"', "'"):
        repaired += '"'  # close the open string

    # Close open arrays
    repaired += "]" * max(0, open_brackets)

    # Close open objects
    repaired += "}" * max(0, open_braces)

    return repaired


# =============================================================================
# FIELD SANITISATION
# =============================================================================

def _sanitise_fields(data: dict) -> dict:
    """
    Coerces each field to its correct type and applies length limits.

    WHY COERCE TYPES?
        The LLM may return confidence as "0.85" (string) instead of 0.85 (float).
        risk_flags may be a single dict instead of a list of dicts.
        actions may be null instead of an empty list.
        Coercing here prevents TypeError crashes deeper in the pipeline.
    """
    # recommendation — must be a non-empty string
    rec = data.get("recommendation", "")
    if not isinstance(rec, str):
        rec = str(rec)
    data["recommendation"] = rec.strip()[:MAX_RECOMMENDATION_LENGTH]

    # reasoning — must be a non-empty string
    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    data["reasoning"] = reasoning.strip()[:MAX_REASONING_LENGTH]

    # confidence — must be float between 0.0 and 1.0
    confidence = data.get("confidence", 0.6)
    try:
        confidence = float(confidence)
        confidence = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, confidence))
    except (TypeError, ValueError):
        logger.warning(f"Invalid confidence value '{confidence}', defaulting to 0.6")
        confidence = 0.6
    data["confidence"] = round(confidence, 2)

    # sowing_advice — string with length limit
    sowing = data.get("sowing_advice", "")
    if not isinstance(sowing, str):
        sowing = str(sowing) if sowing else OPTIONAL_FIELDS_WITH_DEFAULTS["sowing_advice"]
    data["sowing_advice"] = sowing.strip()[:MAX_ADVICE_LENGTH]

    # irrigation_advice — string with length limit
    irrigation = data.get("irrigation_advice", "")
    if not isinstance(irrigation, str):
        irrigation = str(irrigation) if irrigation else OPTIONAL_FIELDS_WITH_DEFAULTS["irrigation_advice"]
    data["irrigation_advice"] = irrigation.strip()[:MAX_ADVICE_LENGTH]

    # market_advice — string with length limit
    market = data.get("market_advice", "")
    if not isinstance(market, str):
        market = str(market) if market else OPTIONAL_FIELDS_WITH_DEFAULTS["market_advice"]
    data["market_advice"] = market.strip()[:MAX_ADVICE_LENGTH]

    # risk_flags — must be a list
    flags = data.get("risk_flags", [])
    if isinstance(flags, dict):
        flags = [flags]  # single dict → wrap in list
    elif not isinstance(flags, list):
        flags = []
    data["risk_flags"] = flags

    # actions — must be a list
    actions = data.get("actions", [])
    if isinstance(actions, dict):
        actions = [actions]
    elif not isinstance(actions, list):
        actions = []
    data["actions"] = actions

    return data


# =============================================================================
# RISK FLAG VALIDATION
# =============================================================================

def _validate_risk_flags(flags: list) -> list:
    """
    Validates and cleans each risk flag in the list.

    EACH RISK FLAG MUST HAVE:
        severity → one of: critical, high, medium, low
        type     → one of: weather, soil, market, timing, pest, water
        message  → non-empty string describing the risk

    INVALID FLAGS:
        Flags missing required fields are given defaults.
        Flags with invalid severity or type are coerced to valid values.
        Flags with empty messages are dropped entirely.
        Duplicate messages (same text) are deduplicated.
        List is capped at MAX_RISK_FLAGS.
    """
    validated = []
    seen_messages = set()

    for flag in flags[:MAX_RISK_FLAGS * 2]:  # allow extra before dedup
        if not isinstance(flag, dict):
            continue

        message = str(flag.get("message", "")).strip()
        if not message:
            continue  # drop empty-message flags

        # Deduplicate by message text
        message_lower = message.lower()
        if message_lower in seen_messages:
            continue
        seen_messages.add(message_lower)

        # Validate severity
        severity = str(flag.get("severity", "medium")).lower()
        if severity not in VALID_SEVERITIES:
            severity = "medium"

        # Validate type
        risk_type = str(flag.get("type", "weather")).lower()
        if risk_type not in VALID_RISK_TYPES:
            risk_type = "weather"

        validated.append({
            "severity": severity,
            "type": risk_type,
            "message": message[:300],  # cap message length
        })

        if len(validated) >= MAX_RISK_FLAGS:
            break

    # Sort by severity: critical first, then high, medium, low
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    validated.sort(key=lambda f: severity_order.get(f["severity"], 4))

    return validated


# =============================================================================
# ACTION VALIDATION
# =============================================================================

def _validate_actions(actions: list) -> list:
    """
    Validates and cleans each action item in the list.

    EACH ACTION MUST HAVE:
        priority  → integer 1–N (1 is most urgent)
        action    → non-empty string describing what to do
        timeframe → string describing when to do it

    INVALID ACTIONS:
        Actions missing required fields are given defaults.
        Priority is re-assigned sequentially if values are missing or duplicate.
        List is capped at MAX_ACTIONS.
    """
    validated = []
    seen_actions = set()

    for i, item in enumerate(actions[:MAX_ACTIONS * 2]):
        if not isinstance(item, dict):
            continue

        action_text = str(item.get("action", "")).strip()
        if not action_text:
            continue

        # Deduplicate by action text
        action_lower = action_text.lower()
        if action_lower in seen_actions:
            continue
        seen_actions.add(action_lower)

        # Priority: use provided value or assign sequentially
        try:
            priority = int(item.get("priority", i + 1))
        except (TypeError, ValueError):
            priority = i + 1

        # Timeframe: default if missing
        timeframe = str(item.get("timeframe", "as soon as possible")).strip()
        if not timeframe:
            timeframe = "as soon as possible"

        validated.append({
            "priority": priority,
            "action": action_text[:300],
            "timeframe": timeframe[:100],
        })

        if len(validated) >= MAX_ACTIONS:
            break

    # Re-sort by priority and reassign sequential priorities
    validated.sort(key=lambda a: a["priority"])
    for i, action in enumerate(validated):
        action["priority"] = i + 1

    return validated


# =============================================================================
# FALLBACK ADVISORY
# =============================================================================

def _build_fallback_advisory(reason: str) -> dict:
    """
    Returns a minimal valid advisory when parsing fails completely.

    WHY NOT RAISE AN EXCEPTION?
        The engine's job is to always return something to the farmer.
        A fallback advisory with low confidence and an honest message
        is better than a 500 error during a live demo.

        The low confidence score (0.3) signals to the frontend that
        this advisory should be displayed with a warning.

    CALLED WHEN:
        - LLM returns None or empty string
        - JSON parsing fails on all strategies
        - Required fields are completely absent
    """
    logger.error(f"Using fallback advisory. Reason: {reason}")

    return {
        "recommendation": (
            "Advisory generation encountered an issue. "
            "Please retry or consult your local agricultural extension officer."
        ),
        "confidence": 0.3,
        "reasoning": (
            f"The advisory engine encountered a parsing error and could not "
            f"generate a complete recommendation. Technical reason: {reason[:200]}"
        ),
        "sowing_advice": OPTIONAL_FIELDS_WITH_DEFAULTS["sowing_advice"],
        "irrigation_advice": OPTIONAL_FIELDS_WITH_DEFAULTS["irrigation_advice"],
        "market_advice": OPTIONAL_FIELDS_WITH_DEFAULTS["market_advice"],
        "risk_flags": [
            {
                "severity": "medium",
                "type": "weather",
                "message": (
                    "Advisory is incomplete due to a system error. "
                    "Do not make critical farming decisions based on this output."
                ),
            }
        ],
        "actions": [
            {
                "priority": 1,
                "action": "Retry the advisory request.",
                "timeframe": "immediately",
            },
            {
                "priority": 2,
                "action": "Contact local Krishi Vigyan Kendra (KVK) for manual advisory.",
                "timeframe": "within 24 hours",
            },
        ],
        "_parse_error": True,
        "_parse_error_reason": reason,
    }


# =============================================================================
# UTILITY — EXTRACT TOOL SUMMARIES FROM REASONING STEPS
# =============================================================================

def extract_tool_summaries(reasoning_steps: list) -> dict:
    """
    Extracts the key data summaries from completed reasoning steps.
    Used by routes.py to populate the advisory's *_summary fields
    in the database without re-running the tools.

    CALLED BY:
        routes.py — after the engine completes, before calling crud.complete_session().
        Pulls weather_summary, crop_summary, soil_summary, market_summary
        from the persisted reasoning steps.

    RETURNS:
        dict with four keys: weather, crop, soil, market.
        Each value is a compact summary dict or None if that tool
        was not called or failed.
    """
    summaries = {
        "weather": None,
        "crop": None,
        "soil": None,
        "market": None,
    }

    for step in reasoning_steps:
        obs = step.get("observation") if isinstance(step, dict) else getattr(step, "observation", None)
        tool = step.get("tool_name") if isinstance(step, dict) else getattr(step, "tool_name", None)

        if not obs or not tool:
            continue

        if tool == "get_weather" and obs.get("status") == "success":
            s = obs.get("summary", {})
            summaries["weather"] = {
                "avg_temp_c":        s.get("avg_temp_c"),
                "total_rainfall_mm": s.get("total_rainfall_mm"),
                "max_humidity_pct":  s.get("max_humidity_percent"),
                "drought_risk":      obs.get("risk_signals", {}).get("drought_stress_risk"),
                "waterlogging_risk": obs.get("risk_signals", {}).get("waterlogging_risk"),
                "irrigation_needed": obs.get("irrigation_signals", {}).get("irrigation_needed"),
                "irrigation_urgency":obs.get("irrigation_signals", {}).get("urgency"),
            }

        elif tool == "get_crop_data" and obs.get("status") == "success":
            sw = obs.get("sowing_window", {})
            wr = obs.get("water_requirements", {})
            summaries["crop"] = {
                "crop":                   obs.get("crop"),
                "category":               obs.get("category"),
                "sowing_status":          sw.get("status"),
                "days_until_close":       sw.get("days_until_close"),
                "days_until_open":        sw.get("days_until_open"),
                "base_irrigation_days":   wr.get("base_irrigation_interval_days"),
                "drought_tolerance":      wr.get("drought_tolerance"),
                "waterlogging_tolerance": wr.get("waterlogging_tolerance"),
            }

        elif tool == "get_soil_profile" and obs.get("status") == "success":
            irr = obs.get("irrigation", {})
            wl  = obs.get("waterlogging", {})
            summaries["soil"] = {
                "soil_type":              obs.get("soil_type"),
                "irrigation_multiplier":  irr.get("irrigation_multiplier"),
                "adjusted_interval_days": irr.get("adjusted_interval_days"),
                "waterlogging_risk":      wl.get("risk_level"),
                "drainage_action":        wl.get("drainage_action_required"),
                "fertility":              obs.get("agronomic_properties", {}).get("fertility"),
            }

        elif tool == "get_market_price" and obs.get("status") == "success":
            sig = obs.get("market_signal", {})
            msp = obs.get("msp_analysis", {})
            mar = obs.get("margin_analysis", {})
            summaries["market"] = {
                "signal":             sig.get("signal"),
                "confidence":         sig.get("confidence"),
                "current_price":      obs.get("current_price", {}).get("national_average_inr_per_quintal"),
                "msp_position":       msp.get("position"),
                "margin_class":       mar.get("classification"),
                "storage_recommended":obs.get("storage_economics", {}).get("recommended"),
            }

    return summaries