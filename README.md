# Vireo Audio — Support Tickets (Set B)

AI-assisted support analytics tool for the Banao Technologies task.

## What it does
- Calculates CSAT, SLA breach, attendance and handle-time metrics per agent.
- Produces a Tier-1 bottom-ten investigation queue.
- Separates Tier-2 Escalations & Warranty from Tier-1 comparison.
- Calculates replacement planning cost using policy unit cost + ₹340 logistics.
- Runs a local text-category model and reports holdout accuracy/error patterns.
- Provides an optional AI-review prompt for deeper qualitative investigation.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app uses the CSVs in `data/`. No API key is required for the core dashboard.

## Important data-quality decision

The supplied extract contains **2,309 legacy tickets with negative handle intervals** (`resolved_at < first_response_at`). The task context says legacy resolution timestamps were reconstructed from an event log, but that event log is not part of this package. The tool therefore excludes negative intervals from valid handle-time statistics rather than inventing a correction.

## Current headline numbers
- Tickets: **11,750**
- CSAT responses: **5,196 / 44.2%**
- Mean CSAT: **3.33 / 5**
- SLA breach rate: **9.1%**
- Replacements: **1,896 / 16.1%**
- Replacement planning cost: **₹34.16 lakh**
- 15% planning target: **~₹5.12 lakh** cost avoided
- Text model holdout accuracy: **77.4%**

## Cost choice

The policy defines replacement planning cost as product unit cost + ₹340 reverse/forward shipping. Finance's rough ₹2,500 estimate conflicts with this policy, so the dashboard uses the policy for the primary calculation and surfaces the discrepancy.

## Deliberate guardrails
- No automatic performance action.
- No Tier-2 vs Tier-1 ranking.
- No automatic replacement decisions.
- No per-ticket paid LLM calls.
