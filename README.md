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
---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/devangdhakaai-create/Agents-League-Hackathon-kisanMind-
cd kisanMind

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env
GITHUB_TOKEN=your_github_pat_with_models_permission
DATABASE_URL=postgresql://kisanmind:password@localhost:5432/kisanmind
ENVIRONMENT=development

# 4. Start PostgreSQL and run
uvicorn app.main:app --reload --port 8000

# 5. Open frontend
# Open index.html in browser
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/advisory` | Run reasoning engine, return full advisory |
| `GET` | `/api/v1/advisory/{id}` | Fetch advisory + reasoning trace |
| `GET` | `/api/v1/sessions` | List recent sessions |
| `GET` | `/api/v1/crops` | Supported crops |
| `GET` | `/api/v1/regions` | Supported regions |
| `GET` | `/api/v1/soils` | Supported soil types |
| `GET` | `/api/v1/health` | Liveness probe |

---

## 🌾 Supported Crops & Regions

**Crops:** Wheat · Rice · Cotton · Maize · Tomato · Soybean

**Regions:** North India · South India · West India · East India · Central India

**Soil Types:** Black Cotton · Loamy · Sandy · Clay · Red Laterite · Sandy Loam

---

## 📊 Example Advisory Output

```json
{
  "recommendation": "Do not sow wheat now — current season is off-window. Use this period for soil preparation and input procurement for November sowing.",
  "confidence": 0.87,
  "sowing_advice": "Wheat sowing window for West India opens November 10. Current status: off_season — 154 days until window opens.",
  "irrigation_advice": "On black cotton soil, irrigate wheat every 29 days (base 21 days × 1.4 multiplier). High waterlogging risk — ensure drainage channels are clear.",
  "market_advice": "Wheat trading at ₹2380/quintal — ₹105 above MSP (4.6% premium). Current signal: SELL NOW. Post-peak seasonal position.",
  "risk_flags": [
    {"severity": "high", "type": "soil", "message": "Black cotton soil has HIGH waterlogging risk. Drainage channel maintenance required before kharif sowing."}
  ],
  "reasoning_trace": [
    {"step": 1, "tool": "get_weather", "thought": "Fetched 7-day weather for (18.52, 73.86): avg temp 31°C, total rainfall 8mm."},
    {"step": 2, "tool": "get_crop_data", "thought": "Retrieved wheat data for west_india: sowing window status = off_season."},
    {"step": 3, "tool": "get_soil_profile", "thought": "Retrieved black cotton profile: multiplier=1.4, adjusted interval=29 days."},
    {"step": 4, "tool": "get_market_price", "thought": "Wheat price ₹2380/q, above MSP, signal=sell_now."}
  ]
}
```

---

## 🏆 Hackathon Track

**Microsoft Agents League 2026 · Reasoning Agents**

Target awards:
- 🥇 Best Reasoning Agent
- 🌍 Hack for Good
- 🎓 Student Award
- 🏅 Best Overall Agent

---

## 👨‍💻 Built By

**Devang Dhaka** — Solo developer

*Built in 7 days for the Microsoft Agents League Hackathon 2026*

---

## 📄 License

MIT License — see [LICENSE](LICENSE)