"""
STEP 5: Checks before acting
----------------------------
Input : moments.csv      (output of step 4, moments.py)
        customers.json   (products held, dismissed offers, contact preference)
Output: decisions.csv    (input for step 6, respond.py)

python decide.py

Rules (the system's judgement):
  1. Financial stress first: a stressed customer gets HELP, never a sale.
     Stress in this month or the 2 months before blocks all sales.
  2. Never offer a product the customer already has.
  3. Never offer a product the customer dismissed in the last 90 days.
  4. Never repeat the same offer within 60 days.
  5. At most one message per customer per month (highest score wins).
"""
import json
import pandas as pd

MOMENTS_FILE = "moments.csv"
CUSTOMERS_FILE = "customers.json"
OUTPUT_FILE = "decisions.csv"

STRESS_MOMENT = "financial_stress"
STRESS_WINDOW_MONTHS = 2
DISMISS_DAYS = 90
REPEAT_DAYS = 60

# type: help = support, sale = commercial, credit = lending (never under stress)
PRODUCTS = {
    "budget_coach":      {"name": "Budget coach with payday forecast (free)", "type": "help"},
    "payment_plan":      {"name": "Talk to KBC about spreading a payment",   "type": "help"},
    "auto_savings":      {"name": "Automatic savings transfer after payday", "type": "sale"},
    "home_insurance":    {"name": "Home insurance for tenants",              "type": "sale"},
    "child_savings":     {"name": "Children's savings account",              "type": "sale"},
    "family_insurance":  {"name": "Family liability insurance",              "type": "sale"},
    "starter_investing": {"name": "Starter investment plan from EUR 25/month", "type": "sale"},
    "mortgage_advice":   {"name": "Mortgage appointment with an advisor",    "type": "credit"},
    "credit_card":       {"name": "Credit card",                             "type": "credit"},
    "personal_loan":     {"name": "Personal loan",                           "type": "credit"},
}

# which products fit which moment, in order of preference
CANDIDATES = {
    "moved_house":      ["home_insurance", "budget_coach"],
    "started_working":  ["auto_savings", "starter_investing"],
    "new_parent":       ["child_savings", "family_insurance"],
    "house_hunting":    ["mortgage_advice"],
    "financial_stress": ["budget_coach", "payment_plan"],
}


def load_customers():
    with open(CUSTOMERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def is_stressed(moments, customer, month):
    """Stress triggered this month or in the previous STRESS_WINDOW_MONTHS months?"""
    m = pd.Period(month, "M")
    s = moments[(moments.customer_id == customer) & (moments.moment == STRESS_MOMENT) & moments.triggered]
    return any(0 <= (m - pd.Period(x, "M")).n <= STRESS_WINDOW_MONTHS for x in s.month)


def pick_product(moment, customer_info, month, stressed, recently_sent=()):
    """Return (product_id, None) or (None, reason why nothing fits)."""
    held = set(customer_info.get("products_held", []))
    month_start = pd.Period(month, "M").to_timestamp()
    dismissed = {d["product"] for d in customer_info.get("dismissed_offers", [])
                 if (month_start - pd.Timestamp(d["date"])).days <= DISMISS_DAYS}
    blocked = []
    for p in CANDIDATES.get(moment, []):
        if p in held:
            blocked.append(f"{p}: already held")
        elif p in dismissed:
            blocked.append(f"{p}: dismissed recently")
        elif p in recently_sent:
            blocked.append(f"{p}: already sent recently")
        elif stressed and PRODUCTS[p]["type"] != "help":
            blocked.append(f"{p}: no sales under financial pressure")
        else:
            return p, None
    return None, "; ".join(blocked) or "no product for this moment"


def decide_all(moments, customers):
    rows = []
    history = {}   # customer -> {product: month sent}
    trig = moments[moments.triggered]
    for (customer, month), group in trig.groupby(["customer_id", "month"]):
        info = customers.get(customer, {"products_held": [], "dismissed_offers": []})
        stressed = is_stressed(moments, customer, month)
        # stress first, then highest score
        group = group.assign(prio=(group.moment != STRESS_MOMENT)).sort_values(["prio", "score"],
                                                                                ascending=[True, False])
        sent = False
        m_now = pd.Period(month, "M")
        recent = {p for p, m in history.get(customer, {}).items()
                  if (m_now.to_timestamp() - pd.Period(m, "M").to_timestamp()).days <= REPEAT_DAYS}
        for r in group.itertuples():
            product, reason = pick_product(r.moment, info, month, stressed, recent)
            if product and sent:
                decision, reason, product = "skip", "max one message per month", None
            elif product:
                decision, sent = "respond", True
                history.setdefault(customer, {})[product] = month
                reason = "customer under financial pressure: help only" if stressed else "passed all checks"
            else:
                decision = "skip"
            rows.append({"customer_id": customer, "month": month, "moment": r.moment, "label": r.label,
                         "score": r.score, "decision": decision, "product": product or "",
                         "product_type": PRODUCTS[product]["type"] if product else "",
                         "stressed": stressed, "reason": reason, "evidence": r.evidence})
    return pd.DataFrame(rows)


def check(decisions, customers):
    resp = decisions[decisions.decision == "respond"]
    print("\nChecks:")
    ahmed_credit = resp[(resp.customer_id == "ahmed") & (resp.product_type == "credit")]
    held_offered = [r for r in resp.itertuples()
                    if r.product in customers.get(r.customer_id, {}).get("products_held", [])]
    stressed_sales = resp[resp.stressed & (resp.product_type != "help")]
    # guard test: pretend Ahmed shows house-hunting while stressed -> must be blocked
    p, why = pick_product("house_hunting", customers["ahmed"], "2026-09", stressed=True)
    tests = [
        ("Ahmed never gets offered credit", ahmed_credit.empty),
        ("Nobody gets offered something they already have", not held_offered),
        ("Stressed customers only get help products", stressed_sales.empty),
        (f"Guard test: stressed Ahmed + mortgage interest -> {p or 'blocked'}", p in (None, "budget_coach", "payment_plan")),
    ]
    for name, ok in tests:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    moments = pd.read_csv(MOMENTS_FILE)
    customers = load_customers()
    decisions = decide_all(moments, customers)
    decisions.to_csv(OUTPUT_FILE, index=False)

    for r in decisions.itertuples():
        what = f"-> {r.product}" if r.decision == "respond" else "SKIP"
        print(f"{r.customer_id:<9} {r.month}  {r.label:<26} {what:<22} ({r.reason})")
    check(decisions, customers)
    print(f"\nSaved: {OUTPUT_FILE}  ({(decisions.decision == 'respond').sum()} messages to send) -> input for step 6")