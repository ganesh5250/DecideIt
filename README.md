# DecideIt Agent — Stage 1 foundation

This repository is being built in stages. Stage 1 establishes the backend-independent
agent interface, bounded planning, run memory, prompt boundaries, and Gemini/Groq
transport. Document, research, analysis, funding and report tools are added in later
stages; this foundation deliberately reports unavailable stages instead of fabricating
results.

## Requirements

- Python 3.10+
- A Gemini or Groq API key for live synthesis (not needed for offline tests)

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows; use cp on macOS/Linux
```

Set `DECIDEIT_LLM_PROVIDER` and the matching API key in `.env`. Load it in your
application with `python-dotenv` or export the variables in your shell.

## Backend integration

```python
from agent import DecideItAgent
from agent.llm import LLMClient

with LLMClient() as llm:
    agent = DecideItAgent(llm=llm)
    result = agent.evaluate({
        "project_title": "Community Plastic Waste Reduction",
        "problem_description": "Plastic waste accumulates in local waterways.",
        "proposed_solution": "A community collection and recycling system.",
        "location": {"country": "India", "state": "Tamil Nadu"},
        "requirements": {"target_beneficiaries": 5000, "timeline_months": 12}
    })
    payload = result.to_dict()  # return payload from a FastAPI route
```

The agent does not implement authentication, persistence, user accounts or database
access. Those remain responsibilities of the DecideIt API backend.

## Offline tests

```bash
python -m unittest discover -s tests -v
```

No API key or network connection is required for these tests.

## Current stage boundaries

- Implemented: provider-neutral LLM wrapper, deterministic workflow planner, bounded
  orchestration, run-local execution history, input validation, JSON serialization.
- Next: common tool protocol and registry; safe document ingestion; search adapters;
  deterministic environmental/social/feasibility/funding evaluators; report generation;
  CLI, sample data, Colab notebook and integration tests.
- No impact score, funding eligibility, citation or measured outcome is fabricated.
# DecideIt
