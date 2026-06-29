# Life Insurance Risk Analysis & Advisory Agent

Agentic AI system that evaluates customer mortality risk, lapse risk, affordability, and coverage needs to produce explainable life insurance recommendations.

## Features

- ML-based mortality and persistency prediction
- Eligibility, coverage, premium, and product recommendation engines
- Google ADK multi-agent coordinator (intake → recommendation → report)
- ADK graph workflow for deterministic structured analysis
- ADK session state, events, and in-memory memory service
- Streamlit UI with chat advisor and structured profile form
- PDF advisory report generation

## Project Structure

```text
backend/
  agent.py                 # ADK root agent entry point
  agents/                  # Coordinator + specialist agents + graph workflow
  app/                     # ADK Runner, session, memory integration
  services/                # ML models, rules engines, report builder
  tools/                   # ADK tools wired to the recommendation pipeline
  orchestrators/           # End-to-end recommendation pipeline
  data/                    # Synthetic training and rules data
  models/                  # Trained joblib models (generated)
streamlit_app.py           # Streamlit frontend
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the environment template and add your Gemini API key:

```bash
copy backend\.env.example backend\.env
```

Set `GOOGLE_API_KEY` in `backend/.env`.

## Run the Application

### Streamlit UI (recommended)

From the `frontend` direrctory:

```bash
streamlit run streamlit_app.py
```

- **Structured Analysis** tab: runs the graph workflow + ML pipeline without Gemini
- **AI Advisor Chat** tab: uses ADK coordinator agent with session state and memory
- **Session State** tab: inspects ADK session artifacts

### ADK Web UI (development)

From the `backend` directory:

```bash
adk web --port 8000
```

Open http://localhost:8000 and select the coordinator agent.

### ADK CLI

```bash
cd backend
adk run .
```

## ADK Architecture

| Component | Purpose |
|-----------|---------|
| `coordinator_agent` | Routes between intake, recommendation, and report specialists |
| `analysis_workflow` | Graph-based deterministic pipeline for structured profiles |
| `InMemorySessionService` | Tracks conversation threads, events, and session state |
| `InMemoryMemoryService` | Stores completed advisory sessions for recall |
| `generate_recommendation_tool` | Runs the full recommendation pipeline |
| `generate_pdf_tool` | Creates downloadable advisory PDF reports |

## Important Notes

- This is an advisory demo system, not a real underwriting engine.
- Chat mode requires a valid `GOOGLE_API_KEY`.
- Structured analysis works offline once models are trained.
- Model files are generated locally and are not committed to source control.

## References

- [ADK Python Quickstart](https://adk.dev/get-started/python/)
- [ADK Graph Workflows](https://adk.dev/graphs/)
- [ADK Workflow Patterns](https://adk.dev/workflows/patterns/)
- [ADK Sessions & State](https://adk.dev/sessions/)
