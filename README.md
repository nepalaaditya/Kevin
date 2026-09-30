# KBC Life Moments Engine (hackathon starter)

pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...      # Windows: set ANTHROPIC_API_KEY=sk-...
streamlit run app.py

No API key? The app still works with rule-based fallback messages.

data.py   -> 3 fake personas (transactions + app events, Apr-Sep 2026)
engine.py -> UNDERSTAND: detect_signals -> detect_moments -> build_profile
             ADAPT: decide_response (LLM, JSON output, guardrails)
app.py    -> Streamlit demo: month slider, message, "Why am I seeing this?",
             editable "What KBC knows about me", Scale tab with cost funnel
