"""
STEP 7: Demo interface
----------------------
Reads the outputs of steps 2-6 and shows the whole story per customer, month by month.

pip install streamlit
streamlit run demo.py
"""
import html
import json
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="KBC Life Moments", page_icon="🔷", layout="wide")

NAVY, BLUE, SOFT = "#003665", "#00AEEF", "#EAF6FC"
MOMENT_ICON = {"moved_house": "🏠", "started_working": "💼", "new_parent": "👶",
               "house_hunting": "🔑", "financial_stress": "🤝"}
CHANNEL_LABEL = {"app_notification": "📱 App notification", "email": "✉️ Email",
                 "kate_chat": "💬 Kate chat", "advisor_call": "📞 Advisor call"}

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.5rem; max-width: 1300px; }}
  h1, h2, h3 {{ color: {NAVY}; }}
  .chip {{ display:inline-block; padding:3px 10px; margin:3px 4px 3px 0; border-radius:14px;
          font-size:0.85rem; border:1px solid #d0d7de; background:#fff; }}
  .chip.appeared {{ border-color:#2e9e5b; color:#1d6b3c; background:#eefaf2; }}
  .chip.disappeared {{ border-color:#c0392b; color:#8e2a20; background:#fdf0ee; }}
  .chip.spiked {{ border-color:#d68910; color:#8a5a07; background:#fff7e8; }}
  .chip.dropped {{ border-color:#7f8c8d; color:#4d5656; background:#f4f6f6; }}
  .chip.app {{ border-color:{BLUE}; color:{NAVY}; background:{SOFT}; }}
  .step {{ font-weight:600; color:{NAVY}; margin:14px 0 6px 0; font-size:1.05rem; }}
  .reason {{ color:#555; font-size:0.9rem; }}
  .phone {{ width:320px; margin:0 auto; border:10px solid #1c1c1e; border-radius:36px;
            background:#f5f7fa; min-height:560px; overflow:hidden; box-shadow:0 10px 30px rgba(0,54,101,.18); }}
  .phone-top {{ background:{NAVY}; color:#fff; padding:18px 16px 14px; }}
  .phone-top .brand {{ font-weight:700; letter-spacing:.5px; font-size:1.1rem; }}
  .phone-top .hi {{ opacity:.85; font-size:.9rem; margin-top:4px; }}
  .balance {{ background:#fff; margin:12px; border-radius:14px; padding:12px 14px; }}
  .balance .lbl {{ color:#667; font-size:.8rem; }}
  .balance .amt {{ color:{NAVY}; font-size:1.5rem; font-weight:700; }}
  .notif {{ background:#fff; margin:12px; border-radius:14px; padding:14px; border-left:5px solid {BLUE}; }}
  .notif.help {{ border-left-color:#2e9e5b; }}
  .notif .ch {{ font-size:.75rem; color:#667; margin-bottom:6px; }}
  .notif .msg {{ color:#1c1c1e; font-size:.95rem; line-height:1.4; }}
  .notif .cta {{ display:inline-block; margin-top:10px; background:{NAVY}; color:#fff; padding:6px 12px;
                 border-radius:18px; font-size:.8rem; }}
  .notif details {{ margin-top:10px; font-size:.8rem; color:#445; }}
  .notif summary {{ cursor:pointer; color:{NAVY}; font-weight:600; }}
  .quiet {{ margin:12px; padding:14px; border-radius:14px; background:#fff; color:#778; font-size:.9rem;
            text-align:center; }}
</style>""", unsafe_allow_html=True)


# ---------------- load pipeline outputs ----------------
def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load():
    tx_file = "transactions_categorised.csv" if os.path.exists("transactions_categorised.csv") \
        else "synthetic_bank_transactions.csv"
    tx = _read(tx_file)
    if "category" not in tx and "true_category" in tx:
        tx = tx.rename(columns={"true_category": "category"})
    tx["date"] = pd.to_datetime(tx["date"])
    tx["month"] = tx["date"].dt.to_period("M").astype(str)
    events = _read("synthetic_app_events.csv")
    if not events.empty:
        events["month"] = pd.to_datetime(events["date"]).dt.to_period("M").astype(str)
    customers = json.load(open("customers.json", encoding="utf-8")) if os.path.exists("customers.json") else {}
    return (tx, events, _read("changes.csv"), _read("moments.csv"), _read("decisions.csv"),
            _read("responses.csv"), customers)


tx, events, changes, moments, decisions, responses, customers = load()
missing = [n for n, d in [("changes.csv", changes), ("moments.csv", moments),
                          ("decisions.csv", decisions), ("responses.csv", responses)] if d.empty]
if tx.empty or missing:
    st.error(f"Missing pipeline output: {', '.join(missing) or 'transactions'}. "
             "Run categorise.py, changes.py, moments.py, decide.py and respond.py first.")
    st.stop()

months = sorted(tx["month"].unique())
cust_ids = sorted(tx["customer_id"].unique())
name_of = lambda c: customers.get(c, {}).get("name", c)
st.session_state.setdefault("rejected", set())


def rows(df, cust, month=None):
    if df.empty:
        return df
    x = df[df.customer_id == cust]
    return x if month is None else x[x.month == month]


# ---------------- header ----------------
st.title("KBC Life Moments")
st.caption("From bank statements to the right help at the right moment, with the customer in control.")

tab_demo, tab_me, tab_scale = st.tabs(["Customer journey", "What KBC knows about me", "Scale to 2.3M"])

with tab_demo:
    c1, c2 = st.columns([1, 3])
    cust = c1.selectbox("Customer", cust_ids, format_func=name_of)
    month = c2.select_slider("Month", options=months, value=months[0],
                             format_func=lambda m: pd.Period(m, "M").strftime("%B %Y"))

    # timeline of the whole journey
    tl = st.columns(len(months))
    for col, m in zip(tl, months):
        trig = rows(moments, cust, m)
        trig = trig[trig.triggered] if not trig.empty else trig
        sent = rows(responses, cust, m)
        icons = "".join(MOMENT_ICON.get(t, "⭐") for t in trig.moment) if len(trig) else "·"
        mark = "✉️" if len(sent) else ""
        style = f"background:{SOFT};border:2px solid {NAVY};" if m == month else "border:1px solid #dde;"
        col.markdown(f"<div style='{style}border-radius:10px;padding:6px;text-align:center;font-size:.8rem'>"
                     f"<b>{pd.Period(m, 'M').strftime('%b')}</b><br>{icons} {mark}</div>",
                     unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        # 1. changes
        st.markdown("<div class='step'>1 · What changed this month</div>", unsafe_allow_html=True)
        ch = rows(changes, cust, month)
        ev = rows(events, cust, month) if not events.empty else pd.DataFrame()
        chips = "".join(
            f"<span class='chip {r.change}'>{html.escape(r.category.replace('_', ' '))} {r.change}"
            + (f" · €{r.now}" if r.change != 'disappeared' else f" · usually €{r.usual}") + "</span>"
            for r in ch.itertuples())
        chips += "".join(f"<span class='chip app'>📱 {html.escape(e.replace('viewed_', '').replace('_', ' '))}</span>"
                         for e in (ev.event.unique() if len(ev) else []))
        st.markdown(chips or "<span class='reason'>Nothing changed compared with this customer's own history.</span>",
                    unsafe_allow_html=True)

        # 2. moments
        st.markdown("<div class='step'>2 · Which life moment could this be?</div>", unsafe_allow_html=True)
        mo = rows(moments, cust, month)
        if mo.empty:
            st.markdown("<span class='reason'>No moment scored. KBC stays quiet.</span>", unsafe_allow_html=True)
        for r in mo.sort_values("score", ascending=False).itertuples():
            status = ("✅ triggered" if r.triggered else
                      "⛔ no must-have signal" if not r.has_must_have else "⏳ below threshold")
            st.markdown(f"**{MOMENT_ICON.get(r.moment, '⭐')} {r.label}** — score {r.score}/{r.threshold} · {status}")
            st.progress(max(0.0, min(r.score / (r.threshold * 1.5), 1.0)))
            with st.expander("Evidence"):
                for e in str(r.evidence).split(" | "):
                    st.write(f"- {e}")

        # 3. decision
        st.markdown("<div class='step'>3 · Should KBC act?</div>", unsafe_allow_html=True)
        de = rows(decisions, cust, month)
        if de.empty:
            st.markdown("<span class='reason'>Nothing to decide.</span>", unsafe_allow_html=True)
        for r in de.itertuples():
            if r.decision == "respond":
                st.success(f"Respond with **{r.product.replace('_', ' ')}** — {r.reason}")
            else:
                st.info(f"Stay quiet on *{r.label}* — {r.reason}")

        # money context
        st.markdown("<div class='step'>Balance so far</div>", unsafe_allow_html=True)
        hist = tx[(tx.customer_id == cust) & (tx.month <= month)].sort_values("date")
        if "balance" in hist:
            st.line_chart(hist.set_index("date")["balance"], height=170)

    with right:
        # 4. the phone
        info = customers.get(cust, {})
        bal = hist["balance"].iloc[-1] if "balance" in hist and len(hist) else None
        resp = rows(responses, cust, month)
        cards = ""
        for r in resp.itertuples():
            if (cust, r.moment) in st.session_state.rejected:
                cards += "<div class='quiet'>Thanks for letting us know. We won't suggest this again.</div>"
                continue
            kind = "help" if r.product in ("budget_coach", "payment_plan") else ""
            cards += (f"<div class='notif {kind}'>"
                      f"<div class='ch'>{CHANNEL_LABEL.get(r.channel, r.channel)} · {html.escape(r.product_name)}</div>"
                      f"<div class='msg'>{html.escape(r.message)}</div>"
                      f"<span class='cta'>{'Get help' if kind else 'Tell me more'}</span>"
                      f"<details><summary>Why am I seeing this?</summary>{html.escape(r.why)}</details></div>")
        if not cards:
            cards = "<div class='quiet'>No message this month.<br>KBC only speaks when it can really help.</div>"
        st.markdown(
            f"<div class='phone'><div class='phone-top'><div class='brand'>KBC Mobile</div>"
            f"<div class='hi'>Hi {html.escape(info.get('name', cust))} 👋</div></div>"
            + (f"<div class='balance'><div class='lbl'>Current account</div>"
               f"<div class='amt'>€ {bal:,.2f}</div></div>" if bal is not None else "")
            + cards + "</div>", unsafe_allow_html=True)

        for r in resp.itertuples():
            if (cust, r.moment) not in st.session_state.rejected:
                if st.button("This isn't right for me", key=f"rej_{cust}_{month}_{r.moment}",
                             width="stretch"):
                    st.session_state.rejected.add((cust, r.moment))
                    st.rerun()
        if len(resp):
            src = ", ".join(sorted(resp.source.unique()))
            st.caption(f"Message written by: {src}")

with tab_me:
    cust2 = st.selectbox("Customer ", cust_ids, format_func=name_of, key="me_cust")
    info = customers.get(cust2, {})
    trig = moments[(moments.customer_id == cust2) & moments.triggered]
    a, b = st.columns(2)
    with a:
        st.subheader(f"{info.get('name', cust2)}, {info.get('age', '')}")
        st.markdown("**Life moments KBC thinks happened**")
        if trig.empty:
            st.write("None detected.")
        for r in trig.drop_duplicates("moment").itertuples():
            rejected = (cust2, r.moment) in st.session_state.rejected
            txt = f"{MOMENT_ICON.get(r.moment, '⭐')} {r.label} (since {pd.Period(r.month, 'M').strftime('%B')})"
            st.markdown(f"~~{txt}~~ *(you corrected this)*" if rejected else txt)
        st.markdown("**Products you have**")
        st.write(", ".join(p.replace("_", " ") for p in info.get("products_held", [])) or "none")
        st.markdown("**Offers you said no to**")
        st.write(", ".join(d["product"].replace("_", " ") for d in info.get("dismissed_offers", [])) or "none")
        st.markdown(f"**How you like to hear from us:** {CHANNEL_LABEL.get(info.get('contact_preference'), '-')}")
    with b:
        st.markdown("**What we never use**")
        st.write("Health data, religion, political views, or anything you didn't agree to share. "
                 "Every message tells you exactly why you got it, and you can correct us at any time.")
        tx_c = tx[tx.customer_id == cust2]
        last = tx_c[tx_c.month == tx_c.month.max()]
        spend = (-last[last.amount < 0].groupby("category")["amount"].sum()).sort_values(ascending=False)
        st.markdown(f"**Where your money went in {pd.Period(tx_c.month.max(), 'M').strftime('%B')}**")
        st.bar_chart(spend, height=260)

with tab_scale:
    st.subheader("From 444 transactions to 6 messages: a funnel, not spam")
    n_msgs = len(responses)
    funnel = pd.DataFrame({
        "stage": ["Transactions", "Unique descriptions", "Changes detected", "Moments scored",
                  "Moments triggered", "Messages sent"],
        "count": [len(tx), tx.description.nunique(), len(changes), len(moments),
                  int(moments.triggered.sum()), n_msgs],
    })
    st.bar_chart(funnel.set_index("stage"), horizontal=True, height=280)
    st.markdown("**Projected to all KBC customers**")
    customers_n = st.number_input("Customers", value=2_300_000, step=100_000)
    rate = n_msgs / (len(cust_ids) * len(months))
    pct = st.slider("Share of customers with a message per month (%)", 1.0, 30.0,
                    round(min(max(rate * 100, 1.0), 30.0), 1))
    cost = st.slider("LLM cost per message (EUR)", 0.0005, 0.02, 0.002, format="%.4f")
    msgs = customers_n * pct / 100
    x, y, z = st.columns(3)
    x.metric("Change detection runs on", f"{customers_n:,.0f} customers")
    y.metric("Messages per month", f"{msgs:,.0f}")
    z.metric("LLM cost per month", f"€ {msgs * cost:,.0f}")
    st.markdown(
        "- **Categorise each merchant once**, not each transaction (our data: "
        f"{tx.description.nunique()} descriptions → far fewer merchants).\n"
        "- **Cheap change detection for everyone**, weighted moment scores, then rule-based checks.\n"
        "- **The LLM only writes words** for the few customers who get a message; rules decide the actions.\n"
        "- **Customer corrections feed back** into the categories and weights.")
