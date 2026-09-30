"""
STEP 6: Personalised response with an LLM
-----------------------------------------
Input : decisions.csv   (output of step 5, decide.py)
        customers.json  (name, contact preference)
        transactions    (for a short money profile)
Output: responses.csv   (input for the demo)

python respond.py            -> LLM, with fallback texts if anything fails
python respond.py --offline  -> fallback texts only (no internet needed)

LLM answers are cached in response_cache.json: after one online run, the demo works offline too.
"""
import json
import os
import sys
import pandas as pd
import re
import llm
from decide import CANDIDATES, PRODUCTS, load_customers

DECISIONS_FILE = "decisions.csv"
CATEGORISED_FILE = "transactions_categorised.csv"
RAW_FILE = "synthetic_bank_transactions.csv"
OUTPUT_FILE = "responses.csv"
CACHE_FILE = "response_cache.json"

CHANNELS = ["app_notification", "email", "kate_chat", "advisor_call"]

PROMPT = f"""You write messages for KBC, a Belgian bank, to one customer.
You get: the detected life moment, the evidence (in plain words), the product KBC decided to offer,
and a short profile.
Rules:
- message: max 2 short sentences, warm, natural, like a helpful person, not a sales pitch.
- The evidence describes money SPENT or RECEIVED. Spending is never savings. Never invent facts.
- Data is monthly: never say "this week", "today" or "yesterday".
- Never write "we noticed", "we saw" or anything that sounds like the bank is watching.
- Do NOT quote the customer's income or balance. Use at most one amount, only if it helps.
- If product type is "help": be calm and supportive. Do NOT mention what they spent money on.
  Focus on what KBC can do for them.
- For family events (baby, moving), use a soft, open tone ("Preparing for a little one?"),
  since the evidence could have another explanation.
- channel: one of {CHANNELS}. Prefer the customer's contact_preference; advisor_call only for
  mortgage advice.
- why: one plain sentence for the "Why am I seeing this?" button, describing what the customer
  did, based only on the evidence.
Return ONLY JSON: {{"message": "...", "channel": "...", "why": "..."}}"""

# used when the LLM is unavailable -> the demo never breaks
FALLBACK = {
    "started_working":  "Congratulations on your first salary, {name}! Want to set aside a small amount automatically the day after payday?",
    "moved_house":      "Settling into your new place, {name}? Tenant insurance can cover your furniture and belongings.",
    "new_parent":       "Preparing for a little one, {name}? A children's savings account lets you start building something for their future.",
    "house_hunting":    "Thinking about buying a home, {name}? An advisor can show you what's realistic for you, no strings attached.",
    "financial_stress": "Money feels tight this month, {name}? Our free budget coach shows what's coming before payday, so there are no surprises.",
}
FALLBACK_BY_PRODUCT = {
    "payment_plan": "If a payment is hard right now, {name}, we can look at spreading it together. No cost, no judgement.",
    "auto_savings": "You're keeping a healthy buffer, {name}. Want to grow it automatically each month after payday?",
}


# ---------- short money profile per customer, up to that month ----------
def load_transactions():
    if os.path.exists(CATEGORISED_FILE):
        df = pd.read_csv(CATEGORISED_FILE)
    else:
        df = pd.read_csv(RAW_FILE).rename(columns={"true_category": "category"})
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    return df


def money_profile(tx, customer, month):
    m = pd.Period(month, "M")
    d = tx[(tx.customer_id == customer) & (tx.month <= m) & (tx.month > m - 3)]
    n = max(d.month.nunique(), 1)
    spend = d[d.amount < 0]
    top = (-spend.groupby("category")["amount"].sum() / n).sort_values(ascending=False).head(3).round()
    return {
        "avg_monthly_income": round(d[d.amount > 0].amount.sum() / n),
        "avg_monthly_spend": round(-spend.amount.sum() / n),
        "current_balance": round(d.sort_values("date").balance.iloc[-1]) if "balance" in d and len(d) else None,
        "top_spending": top.to_dict(),
    }


# ---------- LLM ----------
def get_client():
    try:
        return llm.get_client()
    except Exception as e:
        print(f"LLM not available ({e}) -> using fallback texts")
        return None


def ask_model(client, payload):
    out = llm.ask_json(client, PROMPT + "\n\nInput:\n" + json.dumps(payload, ensure_ascii=False),
                       temperature=0.4)
    if not out.get("message") or out.get("channel") not in CHANNELS:
        raise ValueError(f"bad answer: {out}")
    return out


def humanize(evidence_item):
    """'rent appeared (now EUR 720, usual EUR 0)  (+5)' -> 'a new rent payment of EUR 720'"""
    e = evidence_item.split("  (")[0]
    m = re.match(r"(\w+) (appeared|disappeared|spiked|dropped) \(now EUR (\d+), usual EUR (\d+)\)", e)
    if m:
        cat, change, now, usual = m.groups()
        cat = cat.replace("_", " ")
        income = cat in ("salary", "student job", "child benefit")
        kind = "income" if income else "spending"
        return {"appeared": f"new {kind}: {cat} EUR {now} (none before)",
                "disappeared": f"{cat} {kind} stopped (usually EUR {usual})",
                "spiked": f"higher {cat} {kind}: EUR {now} (usually EUR {usual})",
                "dropped": f"lower {cat} {kind}: EUR {now} (usually EUR {usual})"}[change]
    if e.startswith("app: "):
        event = e[5:].split(" (")[0]
        if "balance" in event:
            return "checking your balance many times in one day"
        return "your visits to the " + event.replace("viewed_", "").replace("_", " ") + " in the app"
    return e


def fallback(row, info):
    first_choice = CANDIDATES.get(row.moment, [None])[0]
    text = (FALLBACK.get(row.moment) if row.product == first_choice else None) \
        or FALLBACK_BY_PRODUCT.get(row.product) \
        or FALLBACK.get(row.moment, "We have a suggestion for you, {name}.")
    channel = "advisor_call" if row.product == "mortgage_advice" else info.get("contact_preference", "app_notification")
    evidence = [humanize(e) for e in str(row.evidence).split(" | ")]
    return {"message": text.format(name=info.get("name", "")), "channel": channel,
            "why": "Based on: " + "; ".join(evidence[:3]) + "."}


# ---------- run ----------
def respond_all(offline=False):
    decisions = pd.read_csv(DECISIONS_FILE)
    todo = decisions[decisions.decision == "respond"]
    customers = load_customers()
    tx = load_transactions()
    cache = json.load(open(CACHE_FILE, encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}
    client = None if offline else get_client()

    rows = []
    for r in todo.itertuples():
        info = customers.get(r.customer_id, {})
        key = f"v2|{r.customer_id}|{r.month}|{r.moment}|{r.product}"
        payload = {
            "customer": {"name": info.get("name"), "age": info.get("age"),
                         "contact_preference": info.get("contact_preference")},
            "moment": r.label,
            "evidence": [humanize(e) for e in str(r.evidence).split(" | ")],
            "product": {"id": r.product, **PRODUCTS[r.product]},
            "profile": money_profile(tx, r.customer_id, r.month),
        }
        if key in cache:
            out, source = cache[key], "llm (cached)"
        elif client:
            try:
                out, source = ask_model(client, payload), f"llm ({llm.PROVIDER})"
                cache[key] = out
                json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  LLM failed for {key}: {e} -> fallback")
                out, source = fallback(r, info), "fallback"
        else:
            out, source = fallback(r, info), "fallback"

        rows.append({"customer_id": r.customer_id, "month": r.month, "moment": r.moment, "label": r.label,
                     "product": r.product, "product_name": PRODUCTS[r.product]["name"],
                     "channel": out["channel"], "message": out["message"], "why": out["why"],
                     "source": source, "evidence": r.evidence})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if not os.path.exists(DECISIONS_FILE):
        raise SystemExit(f"{DECISIONS_FILE} not found. Run step 5 first: python decide.py")
    responses = respond_all(offline="--offline" in sys.argv)
    responses.to_csv(OUTPUT_FILE, index=False)
    for r in responses.itertuples():
        print(f"\n{r.customer_id} {r.month}  [{r.label}]  via {r.channel}  ({r.source})")
        print(f"  KBC: {r.message}")
        print(f"  Why: {r.why}")
    print(f"\nSaved: {OUTPUT_FILE}  ({len(responses)} messages) -> input for the demo")