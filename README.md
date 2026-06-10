# 🌾 KisanMind — Agricultural Reasoning Agent

> **Microsoft Agents League Hackathon 2026 · Reasoning Agents Track**

KisanMind is a multi-signal reasoning agent that helps Indian farmers make better agricultural decisions. It does not guess — it calls four data tools, reasons across all signals simultaneously, and produces a cited, explainable recommendation.

---

## 🎯 What It Does

A farmer submits their crop, location, soil type, and farm size. KisanMind's reasoning engine:

1. **Fetches live weather** → 7-day forecast, ET0 evapotranspiration, irrigation deficit
2. **Loads crop calendar** → sowing window status, critical growth stages, risk factors
3. **Analyses soil profile** → irrigation multiplier, waterlogging risk, crop-soil compatibility
4. **Checks market signals** → current price vs MSP, trend, sell/hold recommendation

Then it reasons across all four signals and produces:

- ✅ A concrete recommendation the farmer can act on **today**
- 💧 Soil-adjusted irrigation schedule (e.g. *"irrigate every 29 days on black cotton soil"*)
- 🌱 Sowing window status with urgency messaging
- 📈 Market advice with current price and MSP position
- ⚠️ Prioritised risk flags (critical → high → medium)
- 📋 Numbered action plan with timeframes
- 🔍 Full reasoning trace — every tool call, every observation, every thought

---

## 🏗️ Architecture
Farmer Form Input
↓
FastAPI Backend
↓
ReasoningEngine (Native ReAct Loop)
↓
┌──────────────────────────────────────┐
│  GPT-4o-mini via GitHub Models API   │
│  Plan → Tool → Observe → Reason      │
└──────────────────────────────────────┘
↓ calls (in order)
┌────────────┬────────────┬────────────┬────────────┐
│  Weather   │    Crop    │    Soil    │   Market   │
│ Open-Meteo │ Static JSON│ Static JSON│ Mocked JSON│
└────────────┴────────────┴────────────┴────────────┘
↓
PostgreSQL (sessions · advisories · reasoning traces · tool cache)
↓
Structured Advisory Response
**Key design decision:** This is a **decision engine**, not a chatbot. The LLM is the reasoning backbone — it decides which tools to call, in what order, and synthesises a multi-signal recommendation. Every reasoning step is persisted to PostgreSQL and returned in the API response.

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11 · FastAPI · Uvicorn |
| Database | PostgreSQL · SQLAlchemy ORM |
| LLM | GPT-4o-mini via GitHub Models API |
| Agent | Native ReAct loop — no LangChain, no CrewAI |
| Weather | Open-Meteo API (free, no key required) |
| Agricultural Data | Curated static JSON datasets |
| Deployment | Azure Container Apps · Azure Database for PostgreSQL |
| Frontend | Vanilla HTML/CSS/JS · Dark UI |

---

## 🧠 Why Native ReAct (No Framework)?

LangChain and CrewAI hide the reasoning. A native ReAct loop exposes it:

```python
# Each iteration:
# 1. Send conversation history to LLM with tool definitions
# 2. LLM returns tool_call (which tool, which args)
# 3. Execute tool → get observation
# 4. Append observation to history
# 5. Repeat until LLM calls final_answer
```

Judges evaluating *reasoning agents* want to see the reasoning — not a framework import.

---

## 📁 Project Structure
kisanmind/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Environment configuration
│   ├── agent/
│   │   ├── engine.py            # ReAct loop — core of the project
│   │   ├── prompts.py           # System prompt + tool definitions
│   │   ├── parser.py            # LLM output validation
│   │   └── tools.py             # Tool registry + dispatcher
│   ├── tools/
│   │   ├── weather.py           # Open-Meteo integration + caching
│   │   ├── crop.py              # Crop calendar tool
│   │   ├── soil.py              # Soil profile tool
│   │   └── market.py            # Market intelligence tool
│   ├── db/
│   │   ├── database.py          # SQLAlchemy engine + sessions
│   │   ├── models.py            # ORM models (4 tables)
│   │   └── crud.py              # All database operations
│   ├── api/
│   │   ├── routes.py            # All API endpoints
│   │   └── schemas.py           # Pydantic request/response models
│   └── data/
│       ├── crops.json           # 6 crops · sowing windows · risk factors
│       ├── soil_profiles.json   # 6 soil types · irrigation multipliers
│       └── market_prices.json   # Commodity prices · MSP · signals
├── Dockerfile
├── docker-compose.yml
└── requirements.txt