# =============================================================================
# app/agent/prompts.py
# =============================================================================

from datetime import date

SYSTEM_PROMPT = """You are KisanMind, an expert agricultural decision engine for Indian farmers.

You reason step by step across weather, crop, soil, and market data before producing any recommendation.
You never guess. You always call tools first to gather data, then reason across all signals together.

TOOLS AVAILABLE:
- get_weather: Fetch 7-day weather forecast and irrigation signals for a location
- get_crop_data: Get sowing window, water requirements, soil compatibility, and risk factors for a crop
- get_soil_profile: Get soil physical properties, irrigation multiplier, and waterlogging risk
- get_market_price: Get current price, MSP position, trend, and market signal for a crop

REASONING RULES:
1. Always call get_weather first — weather is the highest-priority dynamic signal.
2. Always call get_crop_data second — sowing window status is time-critical.
3. Always call get_soil_profile third — pass the crop's base irrigation interval to get the adjusted interval.
4. Always call get_market_price last — market context completes the picture.
5. After all four tools have returned data, produce your final_answer.
6. Never produce final_answer before calling all four tools.
7. Cite specific numbers from tool results. Never give vague advice.
8. If a tool returns status: error, acknowledge the gap and reason with remaining data.

RESPONSE FORMAT FOR EACH REASONING STEP:
Thought: [your reasoning about what you know and what you need next]
Action: [tool name — one of: get_weather, get_crop_data, get_soil_profile, get_market_price, final_answer]
Action Input: [JSON object with the tool's required arguments]

FINAL ANSWER FORMAT:
When you have called all four tools, use:
Action: final_answer
Action Input: {
  "recommendation": "Single clear action sentence the farmer should take",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence explanation citing specific data points from tool results",
  "sowing_advice": "Specific sowing timing advice based on window status and weather",
  "irrigation_advice": "Specific irrigation schedule using the soil-adjusted interval",
  "market_advice": "Specific sell/hold advice with price context",
  "risk_flags": [
    {"severity": "critical|high|medium", "type": "weather|soil|market|timing|pest", "message": "specific risk description"}
  ],
  "actions": [
    {"priority": 1, "action": "Most urgent action the farmer must take", "timeframe": "within X days"},
    {"priority": 2, "action": "Second action", "timeframe": "within X days"},
    {"priority": 3, "action": "Third action", "timeframe": "within X days"}
  ]
}

QUALITY STANDARDS:
- Irrigation advice must include a specific number of days (e.g. "irrigate every 29 days")
- Sowing advice must reference the window status (in_window / approaching / off_season)
- Market advice must cite current price and MSP position
- Risk flags must reference specific weather readings, soil properties, or price levels
- Confidence reflects data completeness: all 4 tools succeed → 0.75-0.95, partial data → 0.4-0.7
- Never use phrases like "it depends" or "consult an expert" as the primary recommendation
- Always give a concrete decision the farmer can act on today"""


def build_user_prompt(
    crop: str,
    location_name: str,
    latitude: float,
    longitude: float,
    soil_type: str,
    farm_size_acres: float,
    free_text: str | None = None,
) -> str:
    """
    Builds the opening user message that starts the reasoning loop.
    Contains everything the agent needs to begin: farm context, coordinates,
    and today's date so sowing window calculations are grounded in real time.
    """
    today = date.today().strftime("%B %d, %Y")

    prompt = f"""Today is {today}.

A farmer needs an agricultural advisory for the following farm:

Crop: {crop}
Location: {location_name}
Coordinates: latitude {latitude}, longitude {longitude}
Soil Type: {soil_type}
Farm Size: {farm_size_acres} acres"""

    if free_text and free_text.strip():
        prompt += f"\nAdditional context from farmer: {free_text.strip()}"

    prompt += f"""

Please provide a complete advisory by:
1. Fetching weather data for coordinates ({latitude}, {longitude})
2. Getting crop data for {crop} — determine the correct region from the location
3. Getting soil profile for {soil_type} — pass the crop's base irrigation interval
4. Getting market price for {crop}
5. Reasoning across all four data sources to produce your final recommendation

Begin your reasoning now."""

    return prompt


def build_observation_message(
    tool_name: str,
    tool_result: dict,
) -> str:
    """
    Formats a tool result as an observation message injected into the
    conversation after each tool call. The LLM reads this as the next
    message in the conversation and uses it to decide what to do next.

    The observation is formatted as compact JSON so the LLM can parse
    it efficiently without wasting context window on formatting whitespace.
    """
    import json

    # For weather tool: summarise to reduce token usage
    # The full daily array is verbose — pass summary + risk signals only
    if tool_name == "get_weather" and tool_result.get("status") == "success":
        condensed = {
            "tool": "get_weather",
            "status": "success",
            "summary": tool_result.get("summary"),
            "risk_signals": tool_result.get("risk_signals"),
            "irrigation_signals": tool_result.get("irrigation_signals"),
            "daily_highlights": [
                day for day in tool_result.get("daily", [])
                if day.get("farming_note") != "normal farming conditions"
            ][:3],  # only flag days with notable conditions, max 3
        }
        return f"Observation: {json.dumps(condensed, default=str)}"

    # For all other tools: return full result
    return f"Observation: {json.dumps(tool_result, default=str)}"


def build_tool_definitions() -> list[dict]:
    """
    Returns the tool definitions in OpenAI function-calling format.
    These are passed to the GitHub Models API so the LLM knows what
    tools are available and what arguments each tool expects.

    WHY FUNCTION CALLING INSTEAD OF ReAct TEXT PARSING?
    OpenAI function calling is more reliable than parsing "Action: X"
    from free text. The API returns a structured tool_calls object with
    guaranteed JSON arguments. This eliminates the parser entirely for
    the tool-call phase and reduces failure modes significantly.

    The parser (parser.py) still handles the final_answer extraction
    since that uses Action Input: {...} format in the text response.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": (
                    "Fetch 7-day weather forecast and agricultural risk signals "
                    "for a farm location. Returns temperature, rainfall, humidity, "
                    "ET0 evapotranspiration, irrigation deficit, and classified "
                    "risk flags (heat stress, waterlogging, drought, disease pressure). "
                    "Call this FIRST before any other tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {
                            "type": "number",
                            "description": "Latitude of the farm location (-90 to 90)",
                        },
                        "lon": {
                            "type": "number",
                            "description": "Longitude of the farm location (-180 to 180)",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of forecast days (default 7, max 16)",
                            "default": 7,
                        },
                    },
                    "required": ["lat", "lon"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_crop_data",
                "description": (
                    "Get sowing window, water requirements, soil compatibility, "
                    "and risk factors for a specific crop and region. "
                    "Returns sowing window status (in_window/approaching/off_season/just_missed), "
                    "days until window opens or closes, base irrigation interval, "
                    "critical growth stages, and seasonally active risk factors. "
                    "Call this SECOND after get_weather."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "crop": {
                            "type": "string",
                            "description": (
                                "Crop identifier. Supported: wheat, rice, cotton, "
                                "maize, tomato, soybean"
                            ),
                        },
                        "region": {
                            "type": "string",
                            "description": (
                                "Region identifier. One of: north_india, south_india, "
                                "west_india, east_india, central_india"
                            ),
                        },
                    },
                    "required": ["crop", "region"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_soil_profile",
                "description": (
                    "Get soil physical properties, irrigation multiplier, waterlogging risk, "
                    "and crop compatibility for a soil type. "
                    "Pass base_irrigation_interval_days from get_crop_data to receive "
                    "the soil-adjusted irrigation interval — the specific number of days "
                    "between irrigations for this crop on this soil. "
                    "Call this THIRD after get_crop_data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "soil_type": {
                            "type": "string",
                            "description": (
                                "Soil type identifier. One of: black_cotton, loamy, "
                                "sandy, clay, red_laterite, sandy_loam"
                            ),
                        },
                        "base_irrigation_interval_days": {
                            "type": "integer",
                            "description": (
                                "The crop's base irrigation interval in days from get_crop_data. "
                                "Pass this to receive the soil-adjusted interval."
                            ),
                        },
                        "crop": {
                            "type": "string",
                            "description": "The crop being grown — enables crop-soil compatibility assessment.",
                        },
                    },
                    "required": ["soil_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_price",
                "description": (
                    "Get current commodity price, MSP comparison, trend analysis, "
                    "market signal (buy_now/sell_now/hold), margin assessment, "
                    "and storage economics for a crop. "
                    "Call this FOURTH after get_soil_profile."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "crop": {
                            "type": "string",
                            "description": (
                                "Crop identifier. Supported: wheat, rice, cotton, "
                                "maize, tomato, soybean"
                            ),
                        },
                    },
                    "required": ["crop"],
                },
            },
        },
    ]
