# =============================================================================
# app/tools/market.py
# =============================================================================
#
# PURPOSE:
#   Loads the market price dataset (market_prices.json) and provides
#   structured commodity market intelligence to the reasoning engine.
#
# ARCHITECTURAL ROLE:
#   Static data tool — market_prices.json is loaded ONCE at module import.
#   In production, this module would be replaced with a live data fetcher
#   (Agmarknet API or NCDEX feed). The interface — get_market_price(crop) →
#   typed dict — stays identical. The reasoning engine never changes.
#   Only this file changes. That is good architecture.
#
# KEY RESPONSIBILITIES:
#   1. Validate requested crop has market data
#   2. Compute MSP comparison and classify price position
#   3. Compute margin assessment from input cost reference
#   4. Generate sell/hold/buy decision with cited rationale
#   5. Flag storage economics: is holding justified by projected price rise?
#   6. Return shaped response the reasoning engine cites directly
#
# PRODUCTION MIGRATION PATH:
#   To replace mocked data with live data:
#   1. Write a new _fetch_live_price(crop) function
#   2. Replace _load_from_static(crop) call with _fetch_live_price(crop)
#   3. Keep all processing functions (_compute_msp_position, etc.) unchanged
#   The reasoning engine, agent, and API are all unaffected.
#   This is the value of the tool abstraction layer.
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

_DATA_PATH = Path(__file__).parent.parent / "data" / "market_prices.json"

try:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        _MARKET_DATA: dict = json.load(f)

    _CROP_MARKETS: dict = _MARKET_DATA.get("market_data", {})
    _SUMMARY_INDEX: dict = _MARKET_DATA.get("market_summary_index", {})

    logger.info(
        f"Market data loaded: {len(_CROP_MARKETS)} crops "
        f"({', '.join(_CROP_MARKETS.keys())})"
    )

except FileNotFoundError:
    logger.critical(f"market_prices.json not found at {_DATA_PATH}. Cannot start.")
    raise
except json.JSONDecodeError as e:
    logger.critical(f"market_prices.json is malformed: {e}. Cannot start.")
    raise


# =============================================================================
# SIGNAL DISPLAY LABELS
# =============================================================================
#
# Convert internal signal codes to human-readable labels.
# The reasoning engine and frontend both use these labels.
#
# =============================================================================

SIGNAL_LABELS = {
    "buy_now":  "BUY / SOW NOW — Strong market incentive to proceed",
    "sell_now": "SELL NOW — Favourable exit point",
    "hold":     "HOLD — Wait for better price window",
    "watch":    "WATCH — Insufficient signal clarity, monitor daily",
}

TREND_LABELS = {
    "rising":  "Rising ↑",
    "stable":  "Stable →",
    "falling": "Falling ↓",
}

VOLATILITY_LABELS = {
    "very_high": "Very High — prices can swing 5–10× in a season",
    "high":      "High — significant intra-season variation",
    "medium":    "Medium — moderate price variation expected",
    "low":       "Low — stable, MSP-backed or inelastic demand",
}


# =============================================================================
# MAIN TOOL FUNCTION
# =============================================================================

def get_market_price(crop: str) -> dict:
    """
    Returns structured market intelligence for the reasoning engine.

    The reasoning engine calls this tool once per session to get:
    - Current wholesale price and MSP comparison
    - 30-day and 90-day price trend
    - Market signal (buy_now / sell_now / hold / watch)
    - Input cost vs current price margin assessment
    - Storage economics: is holding justified?
    - Seasonal forecast and optimal sell window

    ARGUMENTS:
        crop → crop identifier matching a key in market_prices.json
               e.g. "wheat", "rice", "cotton", "maize", "tomato", "soybean"

    RETURNS:
        dict shaped for the reasoning engine. Never raises.

    NORMALISATION:
        crop string is lowercased, stripped, underscores normalised.
    """
    crop = crop.lower().strip().replace(" ", "_").replace("-", "_")

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    if crop not in _CROP_MARKETS:
        return _build_error_response(
            crop=crop,
            error=f"No market data for crop '{crop}'.",
            suggestion=(
                f"Supported crops: {', '.join(sorted(_CROP_MARKETS.keys()))}."
            ),
        )

    raw = _CROP_MARKETS[crop]

    # ------------------------------------------------------------------
    # EXTRACT RAW SECTIONS
    # ------------------------------------------------------------------

    current_prices = raw.get("current_prices", {})
    trend = raw.get("trend_analysis", {})
    signal_data = raw.get("market_signal", {})
    seasonal = raw.get("seasonal_forecast", {})
    input_cost = raw.get("input_cost_reference", {})
    storage = raw.get("storage_advisory", {})
    msp = raw.get("msp")
    msp_note = raw.get("msp_note", "")

    # ------------------------------------------------------------------
    # CURRENT PRICE — NATIONAL AVERAGE
    # ------------------------------------------------------------------
    # Extract the national average price. Field name varies by crop
    # (some crops have variety-specific prices). We take the first
    # numeric value that has "national" or "average" in its key.
    # Fall back to the first numeric value in the dict.
    # ------------------------------------------------------------------

    national_price = _extract_national_price(current_prices)

    # ------------------------------------------------------------------
    # MSP POSITION
    # ------------------------------------------------------------------

    msp_section = _compute_msp_position(
        crop=crop,
        current_price=national_price,
        msp=msp,
        msp_note=msp_note,
        msp_raw=raw.get("price_vs_msp", {}),
    )

    # ------------------------------------------------------------------
    # TREND CLASSIFICATION
    # ------------------------------------------------------------------

    trend_section = _build_trend_section(trend)

    # ------------------------------------------------------------------
    # MARKET SIGNAL
    # ------------------------------------------------------------------

    signal_section = _build_signal_section(
        signal_data=signal_data,
        trend=trend,
        msp_section=msp_section,
    )

    # ------------------------------------------------------------------
    # MARGIN ASSESSMENT
    # ------------------------------------------------------------------

    margin_section = _build_margin_section(
        current_price=national_price,
        input_cost=input_cost,
    )

    # ------------------------------------------------------------------
    # STORAGE ECONOMICS
    # ------------------------------------------------------------------

    storage_section = _build_storage_section(storage)

    # ------------------------------------------------------------------
    # SEASONAL CONTEXT
    # ------------------------------------------------------------------

    seasonal_section = {
        "next_30_days_outlook": seasonal.get("next_30_days"),
        "next_90_days_outlook": seasonal.get("next_90_days"),
        "peak_price_window": seasonal.get("peak_window"),
        "trough_window": seasonal.get("trough_window"),
        "advice_horizon": seasonal.get("advice_horizon"),
        "seasonal_position": trend.get("seasonal_position"),
    }

    # ------------------------------------------------------------------
    # PRICE VOLATILITY
    # ------------------------------------------------------------------

    raw_volatility = raw.get("market", {}).get("price_volatility")
    if not raw_volatility:
        # Some crops store volatility in a nested location
        raw_volatility = "medium"  # safe default

    # ------------------------------------------------------------------
    # ASSEMBLE RESPONSE
    # ------------------------------------------------------------------

    return {
        "tool": "get_market_price",
        "status": "success",
        "crop": crop,
        "data_as_of": raw.get("current_prices", {}).get("as_of", "2026-06-01"),
        "data_note": "Mocked data based on realistic Indian commodity price patterns. Replace with Agmarknet API in production.",

        # Current price data
        "current_price": {
            "national_average_inr_per_quintal": national_price,
            "regional_prices": _extract_regional_prices(current_prices),
            "unit": "INR per quintal (100 kg)",
        },

        # MSP comparison — the primary price reference for Indian farmers
        "msp_analysis": msp_section,

        # Trend — direction and momentum
        "trend": trend_section,

        # Signal — the actionable recommendation
        "market_signal": signal_section,

        # Margin — is the price covering cost of cultivation?
        "margin_analysis": margin_section,

        # Storage — is holding the crop financially justified?
        "storage_economics": storage_section,

        # Seasonal — where are we in the annual price cycle?
        "seasonal_context": seasonal_section,

        # Price volatility profile
        "price_volatility": {
            "level": raw_volatility,
            "description": VOLATILITY_LABELS.get(raw_volatility, raw_volatility),
        },

        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_market_summary(crop: str) -> Optional[dict]:
    """
    Returns the lightweight market summary for a crop from the index.
    Used by the route handler to populate advisory.market_summary
    without loading the full profile.

    CALLED BY: routes.py after advisory is generated.
    """
    crop = crop.lower().strip()
    return _SUMMARY_INDEX.get(crop)


def list_market_crops() -> list[str]:
    """
    Returns the list of crops with market data.
    Used for validation in route handlers.
    """
    return list(_CROP_MARKETS.keys())


def get_quick_signal(crop: str) -> Optional[str]:
    """
    Returns just the market signal string for a crop.
    Convenience function for the agent engine pre-check.
    e.g. "buy_now", "sell_now", "hold"
    """
    crop = crop.lower().strip()
    summary = _SUMMARY_INDEX.get(crop)
    return summary.get("signal") if summary else None


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================

def _extract_national_price(current_prices: dict) -> Optional[float]:
    """
    Extracts the national average price from the current_prices dict.

    Tries keys in priority order:
        1. "national_average"
        2. "national_average_common" (rice has variety splits)
        3. First numeric value that is not the "as_of" date string
    """
    # Priority 1: direct national_average key
    if "national_average" in current_prices:
        val = current_prices["national_average"]
        if isinstance(val, (int, float)):
            return float(val)

    # Priority 2: national_average_common (rice)
    if "national_average_common" in current_prices:
        val = current_prices["national_average_common"]
        if isinstance(val, (int, float)):
            return float(val)

    # Priority 3: first numeric value (not "as_of" string)
    for key, val in current_prices.items():
        if key != "as_of" and key != "unit" and isinstance(val, (int, float)):
            return float(val)

    return None


def _extract_regional_prices(current_prices: dict) -> dict:
    """
    Returns all regional price entries from current_prices,
    excluding metadata keys (unit, as_of).
    """
    excluded = {"as_of", "unit"}
    return {
        key: val
        for key, val in current_prices.items()
        if key not in excluded and isinstance(val, (int, float))
    }


def _compute_msp_position(
    crop: str,
    current_price: Optional[float],
    msp: Optional[float],
    msp_note: str,
    msp_raw: dict,
) -> dict:
    """
    Computes the MSP position and generates the appropriate advisory.

    MSP POSITION LOGIC:
        above_msp  → market price > MSP: sell via open market
        at_msp     → market price within 2% of MSP: both channels viable
        below_msp  → market price < MSP: use government procurement
        no_msp     → crop is not MSP-covered (vegetables): compare to cost only

    WHY THIS MATTERS:
        For MSP-covered crops, the sell channel (open market vs government
        procurement) depends entirely on whether market price is above MSP.
        This is the most important price decision an Indian farmer makes.
        The reasoning engine uses this classification directly.
    """
    if msp is None:
        return {
            "msp": None,
            "msp_covered": False,
            "position": "no_msp",
            "position_label": "No MSP — market-determined price",
            "message": (
                f"{crop.title()} is not covered by MSP. "
                f"Price is entirely market-determined. "
                f"Compare current price against cost of cultivation to assess margin."
            ),
            "sell_channel_recommendation": "Open market only. Monitor wholesale mandi prices weekly.",
        }

    if current_price is None:
        return {
            "msp": msp,
            "msp_covered": True,
            "position": "unknown",
            "position_label": "MSP covered — current price unavailable",
            "message": "MSP data available but current price could not be determined.",
            "sell_channel_recommendation": "Check local mandi price before deciding sell channel.",
        }

    premium = round(current_price - msp, 0)
    premium_pct = round((premium / msp) * 100, 1)

    if premium > msp * 0.02:
        # Market price more than 2% above MSP
        position = "above_msp"
        label = f"ABOVE MSP by ₹{abs(premium)}/quintal (+{abs(premium_pct)}%)"
        message = (
            f"Current market price (₹{current_price}/q) is above MSP (₹{msp}/q). "
            f"Open market sale is more profitable than government procurement."
        )
        sell_channel = (
            "Sell in open mandi market. "
            "Compare your local mandi price with the national average — "
            "if your local price is also above MSP, sell immediately."
        )

    elif premium < -(msp * 0.02):
        # Market price more than 2% below MSP
        position = "below_msp"
        label = f"BELOW MSP by ₹{abs(premium)}/quintal (-{abs(premium_pct)}%)"
        message = (
            f"Current market price (₹{current_price}/q) is BELOW MSP (₹{msp}/q). "
            f"Government procurement at MSP is the better option. "
            f"Contact your nearest APMC or NAFED procurement centre."
        )
        sell_channel = (
            "Use government MSP procurement (PM-AASHA / NAFED). "
            "Do NOT sell in open market at current prices — you will lose money vs MSP."
        )

    else:
        # Within 2% of MSP — at parity
        position = "at_msp"
        label = f"AT MSP — within 2% (₹{premium:+.0f}/quintal)"
        message = (
            f"Market price (₹{current_price}/q) is at parity with MSP (₹{msp}/q). "
            f"Either channel is viable. Check local procurement availability."
        )
        sell_channel = (
            "Either channel is viable at current prices. "
            "If government procurement centre is accessible, prefer it for price certainty. "
            "If mandi access is easier, check local price before deciding."
        )

    return {
        "msp": msp,
        "msp_covered": True,
        "position": position,
        "position_label": label,
        "premium_over_msp_inr": premium,
        "premium_percent": premium_pct,
        "message": message,
        "sell_channel_recommendation": sell_channel,
        "msp_note": msp_note,
    }


def _build_trend_section(trend: dict) -> dict:
    """
    Builds the trend section with human-readable labels and reasoning context.

    The reasoning engine uses `trend_30_day` as the primary signal for
    near-term direction and `trend_90_day` for structural trend.
    When they diverge (e.g. 90-day falling but 30-day stable), the agent
    notes the stabilisation and does not call it a recovery until momentum
    turns positive.
    """
    trend_30 = trend.get("trend_30_day", "stable")
    trend_90 = trend.get("trend_90_day", "stable")
    momentum = trend.get("momentum", "steady")
    seasonal_pos = trend.get("seasonal_position", "unknown")

    # Divergence detection
    divergence_note = None
    if trend_30 == "stable" and trend_90 == "falling":
        divergence_note = (
            "Short-term stabilisation within a longer falling trend. "
            "Not a confirmed reversal — watch for 2 more weeks before concluding recovery."
        )
    elif trend_30 == "rising" and trend_90 == "falling":
        divergence_note = (
            "Short-term recovery within a longer downtrend. "
            "Could be a dead-cat bounce. Confirm with 30-day persistence before acting."
        )
    elif trend_30 == "falling" and trend_90 == "rising":
        divergence_note = (
            "Short-term correction within a longer uptrend. "
            "Likely temporary — structural trend is bullish."
        )

    return {
        "trend_30_day": trend_30,
        "trend_30_day_label": TREND_LABELS.get(trend_30, trend_30),
        "trend_90_day": trend_90,
        "trend_90_day_label": TREND_LABELS.get(trend_90, trend_90),
        "momentum": momentum,
        "price_change_30_day_inr": trend.get("price_change_30_day_inr"),
        "price_change_90_day_inr": trend.get("price_change_90_day_inr"),
        "seasonal_position": seasonal_pos,
        "divergence_note": divergence_note,
        "market_notes": trend.get("notes", ""),
    }


def _build_signal_section(
    signal_data: dict,
    trend: dict,
    msp_section: dict,
) -> dict:
    """
    Builds the market signal section with full context for the reasoning engine.

    The signal (buy_now / sell_now / hold / watch) is the primary output
    the reasoning engine cites in the advisory. The rationale and risk
    are included so the agent can qualify its recommendation appropriately.

    SIGNAL OVERRIDE LOGIC:
        If market signal says "sell_now" but MSP analysis says "below_msp",
        we strengthen the sell_now with the MSP procurement channel guidance.
        If market signal says "hold" but position is "below_msp",
        we override to "sell_now" via government procurement — holding
        below-MSP inventory is economically irrational for most farmers.
    """
    raw_signal = signal_data.get("signal", "watch")
    confidence = signal_data.get("confidence", 0.5)
    rationale = signal_data.get("rationale", "")
    risk_note = signal_data.get("risk_to_signal", "")

    # MSP override: if below MSP and signal is hold → override to sell_now
    msp_position = msp_section.get("position")
    if msp_position == "below_msp" and raw_signal == "hold":
        raw_signal = "sell_now"
        rationale = (
            f"OVERRIDE: Market signal was 'hold' but current price is below MSP. "
            f"Holding below-MSP inventory is not rational for most farmers. "
            f"Original rationale: {rationale}. "
            f"Recommendation: use government MSP procurement."
        )
        confidence = max(confidence, 0.80)

    return {
        "signal": raw_signal,
        "signal_label": SIGNAL_LABELS.get(raw_signal, raw_signal),
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "rationale": rationale,
        "risk_to_signal": risk_note,
        "msp_override_applied": (msp_position == "below_msp" and raw_signal == "sell_now"),
    }


def _build_margin_section(
    current_price: Optional[float],
    input_cost: dict,
) -> dict:
    """
    Computes the gross margin per quintal and classifies it.

    Gross margin = current_price - cost_of_cultivation
    This tells the farmer: at current prices, is this crop worth growing?

    MARGIN CLASSIFICATIONS:
        strong       → margin > 80% of cost (comfortable profit)
        healthy      → margin 50–80% of cost (good return)
        thin         → margin 20–50% of cost (viable but fragile)
        breakeven    → margin 0–20% of cost (minimal profit)
        loss_making  → margin < 0 (price below cost)

    WHY CLASSIFY INSTEAD OF JUST RETURNING THE NUMBER?
        The reasoning engine can read "thin margin — recommend price
        risk management" more effectively than "margin = ₹310/quintal"
        and deciding itself what that implies.
    """
    avg_cost = input_cost.get("avg_cost_of_cultivation_per_quintal")
    raw_assessment = input_cost.get("margin_assessment", "unknown")

    if avg_cost is None or current_price is None:
        return {
            "cost_of_cultivation_per_quintal": None,
            "current_price_per_quintal": current_price,
            "gross_margin_per_quintal": None,
            "margin_ratio": None,
            "classification": "unknown",
            "message": "Insufficient data to compute margin.",
            "notes": input_cost.get("notes", ""),
        }

    gross_margin = round(current_price - avg_cost, 0)
    margin_ratio = round(gross_margin / avg_cost, 2) if avg_cost > 0 else 0

    # Classify margin
    if margin_ratio > 0.80:
        classification = "strong"
        message = (
            f"Strong margin: ₹{gross_margin}/quintal ({int(margin_ratio*100)}% above cost). "
            f"Current prices offer excellent return on investment."
        )
    elif margin_ratio > 0.50:
        classification = "healthy"
        message = (
            f"Healthy margin: ₹{gross_margin}/quintal ({int(margin_ratio*100)}% above cost). "
            f"Good return at current prices."
        )
    elif margin_ratio > 0.20:
        classification = "thin"
        message = (
            f"Thin margin: ₹{gross_margin}/quintal ({int(margin_ratio*100)}% above cost). "
            f"Viable but vulnerable to price drops. Consider forward contracts or FPO selling."
        )
    elif margin_ratio > 0:
        classification = "breakeven"
        message = (
            f"Near-breakeven margin: ₹{gross_margin}/quintal ({int(margin_ratio*100)}% above cost). "
            f"Any price decline will result in a loss. Review input costs and consider crop insurance."
        )
    else:
        classification = "loss_making"
        message = (
            f"Loss-making at current prices: ₹{abs(gross_margin)}/quintal BELOW cost of cultivation. "
            f"Do not sell at current prices. Use MSP procurement if available or hold for recovery."
        )

    return {
        "cost_of_cultivation_per_quintal": avg_cost,
        "current_price_per_quintal": current_price,
        "gross_margin_per_quintal": gross_margin,
        "margin_ratio": margin_ratio,
        "classification": classification,
        "message": message,
        "notes": input_cost.get("notes", ""),
    }


def _build_storage_section(storage: dict) -> dict:
    """
    Evaluates whether storing the harvested crop is economically justified.

    STORAGE ECONOMICS:
        The farmer holds the crop hoping prices rise enough to cover:
        1. Storage cost (Rs/quintal/month × months held)
        2. Risk of further price decline
        3. Quality deterioration risk

        Only recommend storage if projected price increase > storage cost
        with a reasonable confidence.

    The reasoning engine uses `recommended` (bool) and `rationale` to
    include a specific storage action in the advisory.
    """
    recommended = storage.get("recommended", False)
    rationale = storage.get("rationale", "")
    cost_per_month = storage.get("storage_cost_per_quintal_per_month")
    breakeven = storage.get("break_even_price_increase_to_justify_storage")

    # Build economics summary
    if cost_per_month and breakeven:
        economics_note = (
            f"Storage cost: ₹{cost_per_month}/quintal/month. "
            f"Price needs to rise by ₹{breakeven}/quintal to justify 3-month storage. "
            f"{'This rise is projected.' if recommended else 'This rise is NOT projected at current trajectory.'}"
        )
    else:
        economics_note = (
            "Storage not economically applicable for this crop "
            "(perishable or no cost data available)."
        )

    return {
        "recommended": recommended,
        "rationale": rationale,
        "storage_cost_per_quintal_per_month_inr": cost_per_month,
        "breakeven_price_increase_inr": breakeven,
        "economics_note": economics_note,
        "action": (
            "Hold crop in storage — target price recovery window."
            if recommended
            else "Sell at harvest — storage cost not justified by price outlook."
        ),
    }


def _confidence_label(confidence: float) -> str:
    """
    Converts a 0.0–1.0 confidence score to a human-readable label.
    Used in the signal section to help the reasoning engine communicate
    uncertainty levels in the advisory.
    """
    if confidence >= 0.80:
        return "High confidence"
    elif confidence >= 0.65:
        return "Moderate confidence"
    elif confidence >= 0.50:
        return "Low confidence — treat as directional only"
    else:
        return "Very low confidence — insufficient signal clarity"


def _build_error_response(crop: str, error: str, suggestion: str = "") -> dict:
    """
    Returns a structured error dict when market data is unavailable.
    Consistent shape with other tool error responses.
    """
    return {
        "tool": "get_market_price",
        "status": "error",
        "crop": crop,
        "error": error,
        "suggestion": suggestion,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }