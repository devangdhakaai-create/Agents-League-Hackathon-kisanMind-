# =============================================================================
# app/agent/tools.py
# =============================================================================
# PURPOSE: Tool registry and dispatcher for the reasoning engine.
# The engine calls execute_tool(name, args) — this file routes to the
# correct tool function and returns the result.
# =============================================================================

import logging
import time
from typing import Any

# Import all four tool functions built earlier
from app.tools.weather import get_weather
from app.tools.crop import get_crop_data
from app.tools.soil import get_soil_profile
from app.tools.market import get_market_price

# Import prompt builder for tool definitions (OpenAI function format)
from app.agent.prompts import build_tool_definitions

logger = logging.getLogger(__name__)

# =============================================================================
# TOOL REGISTRY
# =============================================================================
# Maps tool name strings to their Python functions.
# The engine uses tool name strings from the LLM's function call.
# This dict is the only place that mapping is defined.
# =============================================================================

TOOL_REGISTRY: dict[str, callable] = {
    "get_weather":      get_weather,       # live API — Open-Meteo
    "get_crop_data":    get_crop_data,     # static JSON — crops.json
    "get_soil_profile": get_soil_profile,  # static JSON — soil_profiles.json
    "get_market_price": get_market_price,  # static JSON — market_prices.json
}

# Names of valid tools the LLM is allowed to call
VALID_TOOL_NAMES = set(TOOL_REGISTRY.keys())

# The sentinel value the LLM uses to signal it is done reasoning
FINAL_ANSWER_TOOL_NAME = "final_answer"


# =============================================================================
# TOOL EXECUTOR
# =============================================================================

def execute_tool(tool_name: str, tool_args: dict) -> tuple[dict, int]:
    """
    Executes a tool by name with given arguments.
    Returns (result_dict, duration_ms).
    Never raises — returns error dict on any failure.

    Called by engine.py once per reasoning loop iteration.
    The duration_ms is persisted to ReasoningStep for performance tracking.
    """
    # Reject unknown tool names — LLM occasionally hallucinates tool names
    if tool_name not in VALID_TOOL_NAMES:
        logger.warning(f"Unknown tool requested: '{tool_name}'")
        return _unknown_tool_error(tool_name), 0

    # Retrieve the function from registry
    tool_fn = TOOL_REGISTRY[tool_name]

    # Record start time for duration tracking
    start_ms = time.time()

    try:
        # Call the tool function with unpacked keyword arguments
        # tool_args is a dict from the LLM's JSON: {"lat": 18.5, "lon": 73.8}
        result = tool_fn(**tool_args)

        # Compute how long the call took in milliseconds
        duration_ms = int((time.time() - start_ms) * 1000)

        logger.info(
            f"Tool '{tool_name}' completed in {duration_ms}ms "
            f"status={result.get('status', 'unknown')}"
        )
        return result, duration_ms

    except TypeError as e:
        # TypeError = wrong arguments passed by LLM (missing required arg,
        # or unexpected kwarg). Log the bad args for debugging.
        duration_ms = int((time.time() - start_ms) * 1000)
        logger.error(f"Tool '{tool_name}' called with bad args {tool_args}: {e}")
        return _argument_error(tool_name, tool_args, str(e)), duration_ms

    except Exception as e:
        # Catch-all for unexpected tool failures (network errors, etc.)
        duration_ms = int((time.time() - start_ms) * 1000)
        logger.error(f"Tool '{tool_name}' raised unexpected error: {e}")
        return _execution_error(tool_name, str(e)), duration_ms


# =============================================================================
# ARGUMENT SANITISATION
# =============================================================================

def sanitise_tool_args(tool_name: str, raw_args: dict) -> dict:
    """
    Cleans and validates tool arguments before execution.
    Handles type coercion for common LLM mistakes:
      - "7" instead of 7 for integer args
      - "18.52" instead of 18.52 for float args
      - extra whitespace in string args

    Called by engine.py before passing args to execute_tool().
    """
    args = dict(raw_args)  # copy — never mutate the original

    if tool_name == "get_weather":
        # lat and lon must be floats — LLM sometimes returns strings
        if "lat" in args:
            args["lat"] = float(args["lat"])
        if "lon" in args:
            args["lon"] = float(args["lon"])
        # days must be int, clamped to valid Open-Meteo range (1–16)
        if "days" in args:
            args["days"] = max(1, min(16, int(args["days"])))

    elif tool_name == "get_crop_data":
        # crop and region must be lowercase strings
        if "crop" in args:
            args["crop"] = str(args["crop"]).lower().strip()
        if "region" in args:
            args["region"] = str(args["region"]).lower().strip()

    elif tool_name == "get_soil_profile":
        # soil_type must be lowercase string
        if "soil_type" in args:
            args["soil_type"] = str(args["soil_type"]).lower().strip()
        # base_irrigation_interval_days must be positive int if present
        if "base_irrigation_interval_days" in args:
            val = args["base_irrigation_interval_days"]
            args["base_irrigation_interval_days"] = max(1, int(val)) if val else None
        # crop must be lowercase string if present
        if "crop" in args:
            args["crop"] = str(args["crop"]).lower().strip()

    elif tool_name == "get_market_price":
        # crop must be lowercase string
        if "crop" in args:
            args["crop"] = str(args["crop"]).lower().strip()

    return args


# =============================================================================
# TOOL CALL TRACKING
# =============================================================================

def get_called_tools(reasoning_steps: list) -> set[str]:
    """
    Returns the set of tool names already called in this session.
    Used by engine.py to detect if all four tools have been called,
    which is the precondition for accepting a final_answer.

    reasoning_steps is the list of step dicts accumulated by the engine.
    Each step dict has a "tool_name" key.
    """
    return {
        step["tool_name"]
        for step in reasoning_steps
        if step.get("tool_name") != FINAL_ANSWER_TOOL_NAME
    }


def all_tools_called(reasoning_steps: list) -> bool:
    """
    Returns True if all four data tools have been called at least once.
    The engine uses this to enforce the rule: no final_answer until
    all four tools have returned data.

    Prevents the LLM from skipping a tool and producing a partial advisory.
    """
    called = get_called_tools(reasoning_steps)
    return VALID_TOOL_NAMES.issubset(called)  # True if all 4 are in called


def get_missing_tools(reasoning_steps: list) -> list[str]:
    """
    Returns list of tool names not yet called.
    Used by engine.py to build a prompt injection when the LLM attempts
    final_answer before calling all tools:
    "You have not yet called: get_market_price. Call it before final_answer."
    """
    called = get_called_tools(reasoning_steps)
    return [t for t in VALID_TOOL_NAMES if t not in called]  # maintain order


# =============================================================================
# ERROR RESPONSE BUILDERS
# =============================================================================
# These return consistently shaped error dicts so the reasoning engine
# always receives a dict with a "status" field it can check.
# Same pattern as the individual tool error responses.
# =============================================================================

def _unknown_tool_error(tool_name: str) -> dict:
    """Error dict for unrecognised tool name."""
    return {
        "tool": tool_name,
        "status": "error",
        "error": f"Tool '{tool_name}' does not exist.",
        "message": (
            f"Available tools: {', '.join(sorted(VALID_TOOL_NAMES))}. "
            f"Use one of these exact names."
        ),
    }


def _argument_error(tool_name: str, args: dict, error: str) -> dict:
    """Error dict for wrong arguments passed to a tool."""
    return {
        "tool": tool_name,
        "status": "error",
        "error": f"Invalid arguments: {error}",
        "provided_args": args,
        "message": (
            f"Check the tool definition for '{tool_name}' and provide "
            f"the correct argument types and names."
        ),
    }


def _execution_error(tool_name: str, error: str) -> dict:
    """Error dict for unexpected runtime failures inside a tool."""
    return {
        "tool": tool_name,
        "status": "error",
        "error": f"Tool execution failed: {error}",
        "message": (
            "Tool encountered an internal error. "
            "Reasoning will continue with data from other tools."
        ),
    }


# =============================================================================
# PUBLIC INTERFACE SUMMARY
# =============================================================================
# engine.py uses these four functions from this file:
#
#   build_tool_definitions()     → pass to GitHub Models API as tools=
#   sanitise_tool_args(name, args) → clean LLM args before execution
#   execute_tool(name, args)     → run the tool, get (result, duration_ms)
#   all_tools_called(steps)      → check if final_answer is allowed
#   get_missing_tools(steps)     → get list of uncalled tools for prompt
# =============================================================================