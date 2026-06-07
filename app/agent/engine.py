# =============================================================================
# app/agent/engine.py
# =============================================================================
# PURPOSE: The ReAct reasoning loop. This is the core of KisanMind.
# Orchestrates: LLM call → tool dispatch → observation → repeat → final answer
# This file is what makes KisanMind a reasoning agent, not a chatbot.
# =============================================================================

import json
import logging
import time
from typing import Optional
from openai import OpenAI  # GitHub Models uses OpenAI-compatible SDK

from app.config import settings
from app.agent.prompts import (
    build_user_prompt,
    build_observation_message,
    build_tool_definitions,
)
from app.agent.tools import (
    execute_tool,
    sanitise_tool_args,
    all_tools_called,
    get_missing_tools,
)
from app.agent.parser import parse_final_answer, extract_tool_summaries
from app.db.database import get_db_context
from app.db import crud

logger = logging.getLogger(__name__)


# =============================================================================
# GITHUB MODELS CLIENT
# =============================================================================
# GitHub Models uses the OpenAI SDK but with a different base_url.
# The API key is your GitHub Personal Access Token.
# This client is module-level — created once, reused for every request.
# =============================================================================

_client = OpenAI(
    base_url=settings.GITHUB_MODELS_BASE_URL,  # "https://models.inference.ai.azure.com"
    api_key=settings.GITHUB_TOKEN,             # GitHub PAT
)


# =============================================================================
# REASONING ENGINE — MAIN CLASS
# =============================================================================

class ReasoningEngine:
    """
    Implements the ReAct loop for KisanMind agricultural advisory.

    REACT PATTERN (per iteration):
      1. Send conversation history to LLM with tool definitions
      2. LLM returns either: tool_call (call a tool) or text (final_answer)
      3. If tool_call: execute tool, append observation, loop
      4. If final_answer: parse, validate, persist, return

    MAX ITERATIONS: settings.MAX_REASONING_STEPS (default 6)
    If the loop hits the cap, the engine forces a final_answer call
    with whatever data has been gathered so far.
    """

    def __init__(self, session_id: str):
        # session_id links all reasoning steps to the FarmerSession row
        self.session_id = session_id

        # Conversation history sent to the LLM on every turn.
        # Starts with system prompt, grows with each tool call + observation.
        self.messages: list[dict] = [
            {"role": "system", "content": settings.SYSTEM_PROMPT
             if hasattr(settings, "SYSTEM_PROMPT") else _get_system_prompt()}
        ]

        # Accumulated reasoning steps — persisted to DB and returned to API
        self.reasoning_steps: list[dict] = []

        # Tool definitions in OpenAI function-calling format
        self.tool_definitions = build_tool_definitions()

        # Step counter
        self.current_step = 0

    def run(
        self,
        crop: str,
        location_name: str,
        latitude: float,
        longitude: float,
        soil_type: str,
        farm_size_acres: float,
        free_text: Optional[str] = None,
    ) -> dict:
        """
        Runs the full ReAct loop for one farmer advisory request.

        RETURNS:
            {
              "advisory": {...},      ← parsed final answer from LLM
              "reasoning_steps": [...], ← all tool calls and observations
              "tool_summaries": {...},  ← compact summaries for DB storage
              "total_duration_ms": int
            }

        Never raises — catches all exceptions and returns a fallback advisory.
        """
        start_time = time.time()

        logger.info(
            f"ReasoningEngine.run() started: session={self.session_id} "
            f"crop={crop} location={location_name}"
        )

        # Build and append the opening user message with farm context
        user_message = build_user_prompt(
            crop=crop,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            soil_type=soil_type,
            farm_size_acres=farm_size_acres,
            free_text=free_text,
        )
        self.messages.append({"role": "user", "content": user_message})

        try:
            # ---- MAIN REACT LOOP ----
            advisory = self._run_loop()

        except Exception as e:
            # If the loop crashes entirely, return a fallback advisory
            logger.error(f"ReasoningEngine loop crashed: {e}", exc_info=True)
            advisory = parse_final_answer(None)  # triggers fallback advisory

        # Extract compact summaries from completed reasoning steps
        tool_summaries = extract_tool_summaries(self.reasoning_steps)

        total_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"ReasoningEngine.run() completed in {total_ms}ms "
            f"steps={len(self.reasoning_steps)}"
        )

        return {
            "advisory":        advisory,
            "reasoning_steps": self.reasoning_steps,
            "tool_summaries":  tool_summaries,
            "total_duration_ms": total_ms,
        }

    # =========================================================================
    # REACT LOOP
    # =========================================================================

    def _run_loop(self) -> dict:
        """
        The inner ReAct loop. Runs until:
          A) LLM calls final_answer and all 4 tools have been called → success
          B) MAX_REASONING_STEPS reached → force final_answer
          C) LLM returns no tool call and no final_answer → force final_answer
        """
        while self.current_step < settings.MAX_REASONING_STEPS:
            self.current_step += 1
            logger.info(f"ReAct step {self.current_step}/{settings.MAX_REASONING_STEPS}")

            # ---- CALL LLM ----
            response = self._call_llm()

            if response is None:
                # LLM call failed — break and return fallback
                logger.error("LLM call returned None — breaking loop")
                break

            # ---- INSPECT RESPONSE ----
            choice = response.choices[0]
            finish_reason = choice.finish_reason  # "tool_calls" | "stop" | "length"
            message = choice.message

            # Append assistant message to conversation history
            # This is critical — the LLM needs full history each turn
            self.messages.append(message)

            # ---- CASE 1: LLM WANTS TO CALL A TOOL ----
            if finish_reason == "tool_calls" and message.tool_calls:
                tool_call = message.tool_calls[0]  # process one tool call per step
                tool_name = tool_call.function.name
                tool_args_raw = tool_call.function.arguments  # JSON string from LLM

                # Parse tool arguments from JSON string
                try:
                    tool_args = json.loads(tool_args_raw)
                except json.JSONDecodeError:
                    logger.warning(f"Bad tool args JSON from LLM: {tool_args_raw}")
                    tool_args = {}

                # Handle final_answer as a tool call (our primary pattern)
                if tool_name == "final_answer":
                    # Check if all 4 tools have been called first
                    if not all_tools_called(self.reasoning_steps):
                        missing = get_missing_tools(self.reasoning_steps)
                        # Inject a correction message and continue the loop
                        self._inject_correction(missing)
                        continue  # go back to top of loop

                    # All tools called — parse and return the final answer
                    logger.info("LLM called final_answer — parsing advisory")
                    return parse_final_answer(tool_args)

                # Regular tool call — execute it
                result, thought = self._execute_and_observe(
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )

                # Continue loop to next LLM call with updated history

            # ---- CASE 2: LLM RETURNED TEXT (no tool call) ----
            elif finish_reason == "stop" and message.content:
                # LLM produced text instead of a tool call.
                # This happens when the LLM decides to reason in text
                # or attempts to write the final_answer as text JSON.
                logger.info("LLM returned text response (no tool call)")

                # Try to extract final_answer JSON from the text
                extracted = self._try_extract_from_text(message.content)
                if extracted and all_tools_called(self.reasoning_steps):
                    return parse_final_answer(extracted)

                # Not a final answer — append as observation and continue
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Continue reasoning. Call the remaining tools: "
                        f"{', '.join(get_missing_tools(self.reasoning_steps))}. "
                        "Use tool calls, not text responses."
                    )
                })

            # ---- CASE 3: TOKEN LIMIT HIT ----
            elif finish_reason == "length":
                # Response was cut off — force completion with what we have
                logger.warning("LLM hit token limit — forcing final_answer")
                break

            else:
                # Unexpected finish_reason — break and force completion
                logger.warning(f"Unexpected finish_reason: {finish_reason}")
                break

        # ---- LOOP ENDED WITHOUT CLEAN FINAL ANSWER ----
        # Either hit MAX_REASONING_STEPS or broke out of loop.
        # Force the LLM to produce a final_answer with current data.
        logger.warning(
            f"Loop ended after {self.current_step} steps without final_answer. "
            f"Forcing completion."
        )
        return self._force_final_answer()

    # =========================================================================
    # LLM CALL
    # =========================================================================

    def _call_llm(self):
        """
        Makes one call to the GitHub Models API with the current
        conversation history and tool definitions.

        Returns the raw API response or None on failure.

        TIMEOUT: settings.LLM_TIMEOUT_SECONDS (default 30s)
        TEMPERATURE: settings.LLM_TEMPERATURE (default 0.2)
        """
        try:
            response = _client.chat.completions.create(
                model=settings.LLM_MODEL,                    # "gpt-4o-mini"
                messages=self.messages,                       # full conversation history
                tools=self.tool_definitions,                  # OpenAI function definitions
                tool_choice="auto",                           # LLM decides when to call tools
                max_tokens=settings.LLM_MAX_TOKENS,          # 1024 per turn
                temperature=settings.LLM_TEMPERATURE,        # 0.2 for consistent reasoning
                timeout=settings.LLM_TIMEOUT_SECONDS,        # 30s timeout
            )
            return response

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None

    # =========================================================================
    # TOOL EXECUTION AND OBSERVATION
    # =========================================================================

    def _execute_and_observe(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict,
    ) -> tuple[dict, str]:
        """
        Executes a tool and appends the result as an observation to
        the conversation history. Also persists the step to PostgreSQL.

        The observation message format is required by OpenAI's API:
        role="tool" with the tool_call_id linking it to the assistant's call.

        Returns (tool_result, thought_text) tuple.
        The thought_text is a synthetic description of what the LLM did
        (we derive it from the tool name since function calling doesn't
        expose a "Thought:" field like text-based ReAct).
        """
        # Sanitise args before execution (type coercion, bounds checking)
        clean_args = sanitise_tool_args(tool_name, tool_args)

        # Execute the tool — always returns (dict, int), never raises
        result, duration_ms = execute_tool(tool_name, clean_args)

        # Build a condensed observation for the conversation history
        observation_content = build_observation_message(tool_name, result)

        # Append tool result to conversation history in OpenAI's required format
        self.messages.append({
            "role": "tool",                        # must be "tool" for function calling
            "tool_call_id": tool_call_id,          # links result to the assistant's call
            "content": observation_content,        # the tool's output
        })

        # Derive a synthetic thought from the tool call context
        thought = _derive_thought(tool_name, clean_args, result)

        # Persist this reasoning step to PostgreSQL
        self._persist_step(
            thought=thought,
            tool_name=tool_name,
            tool_args=clean_args,
            observation=result,
            is_final=False,
            duration_ms=duration_ms,
        )

        return result, thought

    # =========================================================================
    # LOOP CONTROL HELPERS
    # =========================================================================

    def _inject_correction(self, missing_tools: list[str]) -> None:
        """
        Appends a correction message when the LLM attempts final_answer
        before calling all required tools. Forces the loop to continue.

        Missing tools list is presented as a clear instruction so the
        LLM understands exactly what it still needs to do.
        """
        correction = (
            f"You called final_answer too early. "
            f"You must call these tools first: {', '.join(missing_tools)}. "
            f"Call them now before producing your final answer."
        )
        # Append as user message — LLM treats user messages as instructions
        self.messages.append({"role": "user", "content": correction})
        logger.info(f"Injected correction — missing tools: {missing_tools}")

    def _force_final_answer(self) -> dict:
        """
        Forces the LLM to produce a final_answer using data gathered so far.
        Called when the loop hits MAX_REASONING_STEPS or exits unexpectedly.

        Sends a direct instruction with a reduced temperature (0.1) for
        a more deterministic response under time pressure.
        """
        # Tell the LLM to wrap up with what it has
        self.messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum reasoning steps. "
                "Using all the tool data gathered so far, produce your "
                "final_answer immediately. Be concise but complete."
            )
        })

        try:
            # Lower temperature for forced completion — more deterministic
            response = _client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=self.messages,
                tools=self.tool_definitions,
                tool_choice={"type": "function", "function": {"name": "final_answer"}},
                # Force the LLM to call final_answer specifically
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.1,  # lower than normal for forced completion
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )

            choice = response.choices[0]
            if choice.message.tool_calls:
                tool_args_raw = choice.message.tool_calls[0].function.arguments
                tool_args = json.loads(tool_args_raw)
                return parse_final_answer(tool_args)

        except Exception as e:
            logger.error(f"Forced final_answer call failed: {e}")

        # If forced call also fails — return parser fallback
        return parse_final_answer(None)

    def _try_extract_from_text(self, text: str) -> Optional[dict]:
        """
        Attempts to extract a final_answer JSON object from a text response.
        Called when the LLM returns finish_reason="stop" (text) instead of
        a tool call. Some LLMs occasionally write JSON in their text response
        rather than using the function calling interface.
        """
        try:
            # Look for a JSON object in the text
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    # =========================================================================
    # DATABASE PERSISTENCE
    # =========================================================================

    def _persist_step(
        self,
        thought: str,
        tool_name: str,
        tool_args: dict,
        observation: dict,
        is_final: bool,
        duration_ms: int,
    ) -> None:
        """
        Writes one reasoning step to PostgreSQL.
        Called after every tool execution so steps are persisted
        incrementally — even if the engine crashes mid-loop.

        Uses get_db_context() (not FastAPI's Depends) because the engine
        runs outside the HTTP request lifecycle.

        Failures are logged but never raised — DB persistence must not
        interrupt the reasoning loop.
        """
        try:
            with get_db_context() as db:
                crud.create_reasoning_step(
                    db,
                    session_id=self.session_id,
                    step_number=self.current_step,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                    is_final=is_final,
                    duration_ms=duration_ms,
                )
                db.commit()  # commit each step independently
        except Exception as e:
            # Non-fatal — log and continue reasoning
            logger.warning(f"Failed to persist reasoning step {self.current_step}: {e}")

        # Also accumulate in memory for the return value
        self.reasoning_steps.append({
            "step_number": self.current_step,
            "thought":     thought,
            "tool_name":   tool_name,
            "tool_args":   tool_args,
            "observation": observation,
            "is_final":    is_final,
            "duration_ms": duration_ms,
        })


# =============================================================================
# HELPERS
# =============================================================================

def _derive_thought(tool_name: str, tool_args: dict, result: dict) -> str:
    """
    Synthesises a human-readable "thought" string for a tool call.
    Function calling doesn't expose the LLM's reasoning text (unlike
    text-based ReAct), so we generate a descriptive thought from context.
    This thought is displayed in the demo's reasoning trace view.
    """
    status = result.get("status", "unknown")

    # Build descriptive thought per tool
    if tool_name == "get_weather":
        lat = tool_args.get("lat", "?")
        lon = tool_args.get("lon", "?")
        if status == "success":
            summary = result.get("summary", {})
            rain = summary.get("total_rainfall_mm", "?")
            temp = summary.get("avg_temp_c", "?")
            return (
                f"Fetched 7-day weather for ({lat}, {lon}): "
                f"avg temp {temp}°C, total rainfall {rain}mm."
            )
        return f"Attempted weather fetch for ({lat}, {lon}) — status: {status}."

    elif tool_name == "get_crop_data":
        crop = tool_args.get("crop", "?")
        region = tool_args.get("region", "?")
        if status == "success":
            sw = result.get("sowing_window", {})
            sw_status = sw.get("status", "unknown")
            return (
                f"Retrieved crop data for {crop} in {region}: "
                f"sowing window status = {sw_status}."
            )
        return f"Attempted crop data fetch for {crop}/{region} — status: {status}."

    elif tool_name == "get_soil_profile":
        soil = tool_args.get("soil_type", "?")
        if status == "success":
            irr = result.get("irrigation", {})
            adjusted = irr.get("adjusted_interval_days", "?")
            multiplier = irr.get("irrigation_multiplier", "?")
            return (
                f"Retrieved soil profile for {soil}: "
                f"irrigation multiplier={multiplier}, "
                f"adjusted interval={adjusted} days."
            )
        return f"Attempted soil profile fetch for {soil} — status: {status}."

    elif tool_name == "get_market_price":
        crop = tool_args.get("crop", "?")
        if status == "success":
            sig = result.get("market_signal", {}).get("signal", "?")
            price = result.get("current_price", {}).get(
                "national_average_inr_per_quintal", "?"
            )
            return (
                f"Retrieved market data for {crop}: "
                f"price=₹{price}/quintal, signal={sig}."
            )
        return f"Attempted market data fetch for {crop} — status: {status}."

    # Fallback for any other tool name
    return f"Called {tool_name} with args {tool_args} — status: {status}."


def _get_system_prompt() -> str:
    """
    Fallback: imports system prompt directly from prompts.py.
    Used if settings does not have SYSTEM_PROMPT attribute.
    Avoids circular import by importing inside the function.
    """
    from app.agent.prompts import SYSTEM_PROMPT
    return SYSTEM_PROMPT