"""
STEP 2: Automatic categorisation with an LLM
--------------------------------------------
1. Normalise descriptions into merchant keys  ("SPOTIFY P3960" -> "SPOTIFY P")
2. Only ask the LLM about merchants we haven't seen before (cache file)
3. Send them in batches, get JSON back
4. Measure accuracy against true_category

Choose the provider in .env (see llm.py)
python categorise.py
"""
import json
import os
import re
import time
import pandas as pd
import llm   # provider is chosen in .env (groq / ollama / openrouter / mistral)

CACHE_FILE = "category_cache.json"
BATCH_SIZE = 40

CATEGORIES = [
    "salary", "student_job", "child_benefit", "rent", "utilities", "telecom", "groceries",
    "restaurants", "food_delivery", "transport", "car", "travel", "shopping", "furniture",
    "home_improvement", "baby", "subscriptions", "insurance", "savings", "health",
    "cash_withdrawal", "other",
]

PROMPT = f"""You categorise Belgian bank transaction descriptions (Dutch, French or English).
Pick exactly ONE category for each item from this list: {CATEGORIES}

Hints for Belgian statements:
- HUUR / LOYER = rent;  LOON / WEDDE / SALARIS / SALAIRE = salary
- DOMICILIERING / DD = direct debit: look at the company (ENGIE, LUMINUS = utilities; TELENET, ORANGE = telecom)
- GROEIPAKKET / KINDERBIJSLAG = child_benefit;  PAYCONIQ* = payment app, look at the shop after it
- Use "other" only if you really cannot tell.
- STUDENT / STUDENTENJOB / JOBSTUDENT income = student_job (not salary)
- Gyms and fitness (BASIC-FIT, JIMS) = subscriptions, never health
- IKEA, JYSK, LEEN BAKKER, BRICO for furniture stores = furniture / home_improvement
- Fuel stations (SHELL, TOTAL, Q8) = transport

Each item has an id, a description and whether money came IN or went OUT.
Return ONLY JSON like: {{"items": [{{"id": 0, "category": "groceries"}}]}}"""


# ---------- 1. Merchant keys: categorise each merchant once, not each transaction ----------
def merchant_key(description):
    s = description.upper()
    s = re.sub(r"[#*/]", " ", s)       # remove symbols
    s = re.sub(r"\d+", " ", s)         # remove card numbers, store numbers, references
    return re.sub(r"\s+", " ", s).strip()


# ---------- 2. Cache ----------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------- 3. Ask the LLM for one batch ----------
def ask_llm(client, items):
    """items = list of (merchant_key, 'IN'/'OUT'). Returns list of categories, or None if it failed."""
    lines = "\n".join(f'{i}. "{key}" ({direction})' for i, (key, direction) in enumerate(items))
    for attempt in range(3):
        try:
            data = llm.ask_json(client, PROMPT + "\n\nItems:\n" + lines)
            answers = {int(a["id"]): a["category"] for a in data.get("items", [])}
            # anything missing or not in our list becomes "other"
            return [answers.get(i) if answers.get(i) in CATEGORIES else "other" for i in range(len(items))]
        except Exception as e:
            print(f"  LLM error (attempt {attempt + 1}): {str(e)[:200]}")
            time.sleep(10 * (attempt + 1))     # rate limits: wait and retry
    return None


# ---------- 4. Categorise a whole dataframe ----------
def categorise(df, client):
    df = df.copy()
    df["merchant"] = df["description"].map(merchant_key)

    # direction per merchant: did money mostly come in or go out?
    direction = df.groupby("merchant")["amount"].mean().map(lambda x: "IN" if x > 0 else "OUT")

    cache = load_cache()
    new = [m for m in direction.index if m not in cache]
    print(f"{df['description'].nunique()} descriptions -> {len(direction)} merchants, "
          f"{len(new)} new to ask the LLM")

    for start in range(0, len(new), BATCH_SIZE):
        batch = new[start:start + BATCH_SIZE]
        print(f"  asking {llm.PROVIDER} about {len(batch)} merchants...")
        cats = ask_llm(client, [(m, direction[m]) for m in batch])
        if cats is None:
            print("  Batch failed, NOT cached. Fix the key/provider and run again.")
            continue
        cache.update(dict(zip(batch, cats)))
        save_cache(cache)                        # save after every batch

    df["category"] = df["merchant"].map(cache)
    return df


# ---------- 5. How good is it? ----------
def evaluate(df):
    correct = (df["category"] == df["true_category"]).mean()
    print(f"\nAccuracy: {correct:.1%} of transactions categorised correctly")
    wrong = df[df["category"] != df["true_category"]]
    if not wrong.empty:
        print("\nMistakes (merchant -> predicted / true):")
        summary = wrong.groupby(["merchant", "category", "true_category"]).size()
        print(summary.to_string())


def client_from_env():
    return llm.get_client()


if __name__ == "__main__":
    client = client_from_env()
    df = pd.read_csv("synthetic_bank_transactions.csv")

    # IMPORTANT: the LLM only sees description + amount. true_category / story_tag are for checking only.
    result = categorise(df[["transaction_id", "customer_id", "date", "description", "amount",
                            "balance", "true_category"]], client)
    evaluate(result)
    result.drop(columns=["true_category"]).to_csv("transactions_categorised.csv", index=False)
    print("\nSaved: transactions_categorised.csv")
