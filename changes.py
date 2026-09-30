"""
STEP 3: Change detection
------------------------
Input : transactions_categorised.csv   (output of step 2, categorise.py)
Output: changes.csv                    (input for step 4, moment scoring)

python changes.py            -> uses Gemini categories from step 2
python changes.py --true     -> uses true_category (test without Gemini / while step 2 is being built)
"""
import os
import sys
import pandas as pd

CATEGORISED_FILE = "transactions_categorised.csv"   # from step 2
RAW_FILE = "synthetic_bank_transactions.csv"         # original fake data
OUTPUT_FILE = "changes.csv"                          # for step 4


# ---------- Load the output of step 2 ----------
def load_transactions(use_true=False):
    if use_true:
        print(f"Using true_category from {RAW_FILE} (test mode)")
        df = pd.read_csv(RAW_FILE).rename(columns={"true_category": "category"})
    elif os.path.exists(CATEGORISED_FILE):
        print(f"Using Gemini categories from {CATEGORISED_FILE}")
        df = pd.read_csv(CATEGORISED_FILE)
    else:
        print(f"{CATEGORISED_FILE} not found -> running step 2 first...")
        from categorise import categorise, client_from_env
        df = categorise(pd.read_csv(RAW_FILE), client_from_env())
        df.drop(columns=["true_category"], errors="ignore").to_csv(CATEGORISED_FILE, index=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------- The function from before ----------
def detect_changes(df, customer_id, month, spike=1.5, drop=0.5):
    """Compare one customer's month with their OWN history, per category."""
    month = pd.Period(month, "M")
    d = df[df["customer_id"] == customer_id].copy()
    d["month"] = d["date"].dt.to_period("M")

    totals = d.groupby(["month", "category"])["amount"].sum().abs().unstack(fill_value=0)
    past = totals[totals.index < month]
    now = totals.loc[month] if month in totals.index else pd.Series(0, index=totals.columns)
    if past.empty:
        return []

    changes = []
    for cat in totals.columns:
        history = past[cat]
        active = history[history > 0]
        usual = active.tail(3).median() if len(active) else 0
        current = now[cat]
        regular = len(active) >= 2

        if usual == 0 and current > 0:
            change = "appeared"
        elif regular and current == 0 and history.iloc[-1] > 0:
            change = "disappeared"
        elif regular and current > spike * usual:
            change = "spiked"
        elif len(active) >= 3 and 0 < current < drop * usual:   # stopped = "disappeared", not "dropped"
            change = "dropped"
        else:
            continue
        changes.append({"category": cat, "change": change,
                        "now": round(current), "usual": round(usual)})
    return changes


# ---------- Run it for every customer and every month ----------
def detect_all(df):
    months = sorted(df["date"].dt.to_period("M").unique())
    rows = []
    for customer in sorted(df["customer_id"].unique()):
        for month in months:
            for ch in detect_changes(df, customer, month):
                rows.append({"customer_id": customer, "month": str(month), **ch})
    return pd.DataFrame(rows, columns=["customer_id", "month", "category", "change", "now", "usual"])


# ---------- "Done when" checks ----------
def check(changes):
    def has(cust, cat, change):
        return not changes[(changes.customer_id == cust) & (changes.category == cat)
                           & (changes.change == change)].empty

    print("\nChecks:")
    tests = [
        ("Lisa: rent appeared", has("lisa", "rent", "appeared")),
        ("Ahmed: salary disappeared", has("ahmed", "salary", "disappeared")),
    ]
    if "marc" in changes.customer_id.values or "marc" in set(changes.customer_id):
        marc = changes[changes.customer_id == "marc"]
        tests.append(("Marc: only furniture appeared",
                      set(zip(marc.category, marc.change)) == {("furniture", "appeared")}))
    else:
        tests.append(("Marc: only furniture appeared", None))
    for name, ok in tests:
        print(f"  {'OK  ' if ok else 'MISSING' if ok is None else 'FAIL'}  {name}")


if __name__ == "__main__":
    df = load_transactions(use_true="--true" in sys.argv)
    changes = detect_all(df)
    changes.to_csv(OUTPUT_FILE, index=False)

    for customer, group in changes.groupby("customer_id"):
        print(f"\n=== {customer} ===")
        print(group.drop(columns="customer_id").to_string(index=False))
    check(changes)
    print(f"\nSaved: {OUTPUT_FILE}  ({len(changes)} changes) -> input for step 4")
