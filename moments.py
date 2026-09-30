"""
STEP 4: Moment scoring
----------------------
Input : changes.csv                      (output of step 3, changes.py)
        synthetic_app_events.csv         (app behaviour, optional)
        synthetic_bank_transactions.csv  (only for the balance)
Output: moments.csv                      (input for step 5, checks before acting)

python moments.py

How it works:
  every change gets points per moment (weights), negative points make a moment LESS likely,
  a moment needs at least one must-have change, and only counts above its threshold.
"""
import os
import pandas as pd

CHANGES_FILE = "changes.csv"
EVENTS_FILE = "synthetic_app_events.csv"
RAW_FILE = "synthetic_bank_transactions.csv"
OUTPUT_FILE = "moments.csv"

# ---------------- THE WEIGHT TABLES (your idea) ----------------
# key = "category change"  (from step 3)  or  "app event" / "balance negative" (extra signals)
MOMENTS = {
    "moved_house": {
        "label": "Moved into a new home",
        "must_have": ["rent appeared"],
        "threshold": 7,
        "weights": {
            "rent appeared": 5,
            "utilities appeared": 3,
            "furniture appeared": 2,
            "home_improvement appeared": 1,
            "app viewed_home_insurance_page": 2,
            "salary disappeared": -3,          # losing income makes a move less likely
        },
    },
    "started_working": {
        "label": "Started first job",
        "must_have": ["salary appeared"],
        "threshold": 5,
        "weights": {
            "salary appeared": 5,
            "student_job disappeared": 3,
        },
    },
    "new_parent": {
        "label": "Became a parent",
        "must_have": ["baby appeared"],
        "threshold": 5,
        "weights": {
            "baby appeared": 5,
            "child_benefit appeared": 4,
            "furniture appeared": 1,
        },
    },
    "house_hunting": {
        "label": "Looking to buy a home",
        "must_have": ["app viewed_mortgage_page"],
        "threshold": 5,
        "weights": {
            "app viewed_mortgage_page": 5,
            "savings spiked": 1,
            "salary disappeared": -5,
            "balance negative": -5,
        },
    },
    "financial_stress": {                      # protective moment: help, never sell
        "label": "Under financial pressure",
        "must_have": ["salary disappeared", "balance negative", "food_delivery spiked",
                      "shopping spiked", "restaurants spiked"],   # any one of these
        "threshold": 6,
        "weights": {
            "salary disappeared": 5,
            "balance negative": 5,
            "app viewed_balance_5x_in_one_day": 3,
            "food_delivery spiked": 2,
            "shopping spiked": 2,
            "restaurants spiked": 2,
            "savings spiked": -3,
        },
    },
}


# ---------------- Load: turn everything into "signals" per customer per month ----------------
def load_signals():
    """Returns {(customer, month): {signal_key: description}}"""
    signals = {}

    def add(customer, month, key, text):
        signals.setdefault((customer, month), {})[key] = text

    # 1. changes from step 3
    changes = pd.read_csv(CHANGES_FILE)
    for r in changes.itertuples():
        add(r.customer_id, r.month, f"{r.category} {r.change}",
            f"{r.category} {r.change} (now EUR {r.now}, usual EUR {r.usual})")

    # 2. app events (intent)
    if os.path.exists(EVENTS_FILE):
        ev = pd.read_csv(EVENTS_FILE)
        ev["month"] = pd.to_datetime(ev["date"]).dt.to_period("M").astype(str)
        for (cust, month, event), n in ev.groupby(["customer_id", "month", "event"]).size().items():
            add(cust, month, f"app {event}", f"app: {event} ({n}x)")

    # 3. balance below zero
    if os.path.exists(RAW_FILE):
        tx = pd.read_csv(RAW_FILE)
        tx["month"] = pd.to_datetime(tx["date"]).dt.to_period("M").astype(str)
        low = tx.groupby(["customer_id", "month"])["balance"].min()
        for (cust, month), bal in low[low < 0].items():
            add(cust, month, "balance negative", f"balance went negative (EUR {bal:.0f})")

    return signals


# ---------------- Score one customer-month against every moment ----------------
def score(present):
    """present = {signal_key: description}. Returns list of scored moments."""
    results = []
    for moment_id, m in MOMENTS.items():
        evidence = [f"{present[k]}  ({w:+d})" for k, w in m["weights"].items() if k in present]
        if not evidence:
            continue
        total = sum(w for k, w in m["weights"].items() if k in present)
        has_must = any(k in present for k in m["must_have"])
        results.append({
            "moment": moment_id,
            "label": m["label"],
            "score": total,
            "threshold": m["threshold"],
            "has_must_have": has_must,
            "triggered": has_must and total >= m["threshold"],
            "evidence": " | ".join(evidence),
        })
    return results


def score_all():
    rows = []
    for (customer, month), present in sorted(load_signals().items()):
        for r in score(present):
            rows.append({"customer_id": customer, "month": month, **r})
    cols = ["customer_id", "month", "moment", "label", "score", "threshold",
            "has_must_have", "triggered", "evidence"]
    return pd.DataFrame(rows, columns=cols)


# ---------------- "Done when" checks ----------------
def check(df):
    def triggered(cust, moment, month=None):
        x = df[(df.customer_id == cust) & (df.moment == moment) & df.triggered]
        return not (x if month is None else x[x.month == month]).empty

    print("\nChecks:")
    tests = [
        ("Lisa: moved_house triggered in 2026-07", triggered("lisa", "moved_house", "2026-07")),
        ("Lisa: started_working triggered in 2026-06", triggered("lisa", "started_working", "2026-06")),
        ("Tom & Sara: new_parent triggered", triggered("tom_sara", "new_parent")),
        ("Ahmed: financial_stress triggered", triggered("ahmed", "financial_stress")),
        ("Marc: moved_house NOT triggered (sofa only)",
         None if "marc" not in set(df.customer_id) else not triggered("marc", "moved_house")),
    ]
    for name, ok in tests:
        print(f"  {'OK     ' if ok else 'MISSING' if ok is None else 'FAIL   '}  {name}")


if __name__ == "__main__":
    if not os.path.exists(CHANGES_FILE):
        raise SystemExit(f"{CHANGES_FILE} not found. Run step 3 first: python changes.py")
    df = score_all()
    df.to_csv(OUTPUT_FILE, index=False)

    for customer, g in df.groupby("customer_id"):
        print(f"\n=== {customer} ===")
        for r in g.itertuples():
            status = "TRIGGERED " if r.triggered else ("no must-have" if not r.has_must_have else "below     ")
            print(f"{r.month}  {status}  {r.label:<26} {r.score:>3}/{r.threshold}   {r.evidence}")
    check(df)
    print(f"\nSaved: {OUTPUT_FILE}  ({df.triggered.sum()} triggered moments) -> input for step 5")
