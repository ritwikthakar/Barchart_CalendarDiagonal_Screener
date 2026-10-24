"""Streamlit UI: Barchart calendar / diagonal play screener (SPCX-style + NFLX-style + earnings plays).
Run:  streamlit run app.py
"""
import glob, os
import pandas as pd
import streamlit as st
import screener_core as sc

st.set_page_config(page_title="Options Play Screener", layout="wide")
DEFAULT_DIR = os.environ.get("BARCHART_DIR", "./data")

# ------------------------------------------------------------------ sidebar: data
st.sidebar.header("Data")
ups = st.sidebar.file_uploader("Barchart screener CSVs (all 6 at once)", type="csv", accept_multiple_files=True)
sources = [(u, u.name) for u in ups]


def _open(s):
    import io
    return io.BytesIO(s.getvalue()) if hasattr(s, "getvalue") else s
if not sources and os.path.isdir(DEFAULT_DIR):
    paths = glob.glob(os.path.join(DEFAULT_DIR, "*.csv"))
    if paths:
        latest = max(sc.file_date(p) or pd.Timestamp.min for p in paths)
        sources = [(p, os.path.basename(p)) for p in paths if sc.file_date(p) == latest]
        st.sidebar.caption(f"Loaded {len(sources)} file(s) from {DEFAULT_DIR}")
if not sources:
    st.title("Options Play Screener")
    st.info("Upload the Barchart calendar and diagonal screener CSVs in the sidebar to begin.")
    st.stop()

earn_src = [s for s in sources if sc.is_earnings_file(_open(s[0]))]
sources = [s for s in sources if not sc.is_earnings_file(_open(s[0]))]
file_events = sc.parse_earnings_file(_open(earn_src[0][0])) if earn_src else {}
if not sources:
    st.warning("Only an earnings file was found - add the Barchart calendar/diagonal CSVs."); st.stop()

dates = [d for d in (sc.file_date(n) for _, n in sources) if d is not None]
asof = st.sidebar.date_input("As-of date (DTE reference)", value=(max(dates) if dates else pd.Timestamp.today()).date())
asof = pd.Timestamp(asof)

st.sidebar.header("Earnings")
if file_events:
    st.sidebar.success(f"Earnings file: {len(file_events)} reports in the next 7 trading days ({earn_src[0][1][:40]})")
else:
    st.sidebar.caption("No earnings file found - using the built-in fallback list. Upload Barchart's "
                       "'earnings in next 7 trading days' CSV for the full calendar.")
ev_txt = st.sidebar.text_area("Manual overrides: SYMBOL YYYY-MM-DD AMC|BMO (win over the file)",
                              "" if file_events else sc.DEFAULT_EVENTS, height=120,
                              key="ev_file" if file_events else "ev_default")
rec_txt = st.sidebar.text_area("Already reported (post-event): same format", sc.DEFAULT_RECENT, height=100)
events = {**file_events, **sc.parse_events(ev_txt)}
recent = set(sc.parse_events(rec_txt))


@st.cache_data(show_spinner=False)
def load_all(blobs, asof_str):
    frames, notes = [], []
    for name, data in blobs:
        import io
        try:
            f = sc.load_barchart(io.BytesIO(data), pd.Timestamp(asof_str), label=name)
            frames.append(f); notes.append((name, len(f), f.Kind.iloc[0] if len(f) else "?",
                                            f.CP.iloc[0] if len(f) else "?", f.Struct.mode().iat[0] if len(f) else "?"))
        except Exception as e:  # keep going if one file is odd
            notes.append((name, 0, "error", str(e)[:60], ""))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), notes


blobs = tuple((n, (s.getvalue() if hasattr(s, "getvalue") else open(s, "rb").read())) for s, n in sources)
raw, notes = load_all(blobs, str(asof.date()))
if raw.empty:
    st.error("No usable rows found."); st.stop()
df = sc.tag_events(raw, events, recent)

# ------------------------------------------------------------------ shared model controls
no_etf = st.sidebar.checkbox("Exclude index/sector ETFs", True)
if no_etf:
    df = df[~df.Symbol.isin(sc.ETFS)]

st.sidebar.header("Model (expected value)")
calm_mult = st.sidebar.slider("Calm scenario: realised vol as fraction of back-leg IV", 0.5, 1.0, 0.85, 0.05)
calm_scale = st.sidebar.slider("Calm scenario (earnings): event move as fraction of implied", 0.3, 1.0, 0.7, 0.05)
st.sidebar.caption("Base scenario = realised vol equals implied. EV is model output, not a forecast. "
                   "Exit at the short-leg expiry, long leg sold ~3% under model mid, flat back IV.")

COLS = ["Symbol", "Kind", "CP", "Px", "KS", "KL", "ExpS", "ExpL", "DTE_S", "Net", "IVS", "IVL", "Skew",
        "IVR", "IVHV", "Delta", "Vega"]
EVC = ["EV_calm%", "PoP_calm%", "EV_base%", "PoP_base%"]
FMT = {c: st.column_config.NumberColumn(c, format="%.1f") for c in ["Skew", "IVS", "IVL", "IVR", "EV_calm%", "EV_base%", "PoP_calm%", "PoP_base%"]}
FMT.update({"Net": st.column_config.NumberColumn("Debit", format="%.2f"), "Px": st.column_config.NumberColumn(format="%.2f"),
            "Delta": st.column_config.NumberColumn(format="%.3f"), "Vega": st.column_config.NumberColumn(format="%.3f"),
            "IVHV": st.column_config.NumberColumn("IV/HV", format="%.2f"),
            "ExpS": st.column_config.DateColumn("Short exp"), "ExpL": st.column_config.DateColumn("Long exp"),
            "DTE_S": "Short DTE"})


def show(res, sort_by, drift=0.0, key="t", max_ev=400):
    """Run the EV model on the shortlist, show the table and a trade card."""
    if res.empty:
        st.warning("No rows pass these filters - loosen a threshold."); return
    res = res.sort_values("Skew", ascending=False).head(max_ev)
    res = sc.add_ev(res, calm_mult, calm_scale, drift)
    res = res.sort_values(sort_by, ascending=False).reset_index(drop=True)
    tag = res.EventFront.map({True: "event in front", False: ""}) + res.Reported.map({True: " reported", False: ""})
    view = res.assign(Note=tag)[COLS + EVC + ["Note"]]
    st.caption(f"{len(res)} candidates, sorted by {sort_by}")
    st.dataframe(view, column_config=FMT, use_container_width=True, hide_index=True, height=420)
    st.download_button("Download CSV", view.to_csv(index=False), f"{key}_candidates.csv", key=f"dl_{key}")
    trade_card(res, key)


def trade_card(res, key):
    st.subheader("Trade card")
    labels = [f"{r.Symbol} {r.CP} {r.KL:g}/{r.KS:g} {r.ExpS:%m/%d}->{r.ExpL:%m/%d} @ {r.Net:.2f}" for r in res.itertuples()]
    i = st.selectbox("Pick a candidate", range(len(res)), format_func=lambda j: labels[j], key=f"pick_{key}")
    r = res.iloc[i]
    n = st.number_input("Contracts", 1, 100, 2, key=f"n_{key}")
    tgt = st.number_input("Exit target (% of debit)", 5.0, 100.0, 13.0, 1.0, key=f"tg_{key}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost", f"${r.Net * 100 * n:,.0f}")
    c2.metric("GTC target (net)", f"{r.Net * (1 + tgt / 100):.2f}")
    c3.metric("Target gain", f"${r.Net * tgt / 100 * 100 * n:,.0f}")
    c4.metric("Short-leg gap (IV pts)", f"{r.Skew:+.1f}")
    pc = sc.payoff_curve(r)
    st.line_chart(pc, height=240)
    be = pc[pc["P/L per share"] > 0].index
    if len(be):
        st.caption(f"Break-evens at short expiry (flat IV, no event): ~{be.min():.1f} to ~{be.max():.1f} (spot {r.Px:.2f}).")
    if r.EventFront:
        st.warning(f"Earnings {r.EventDate:%m/%d} lands inside the short leg - the payoff chart ignores the event jump.")
    st.info("Close before the short leg's expiry day (~3:30pm ET) to avoid pin/assignment risk on a short ITM option.")


tab_data, tab_spcx, tab_nflx, tab_earn = st.tabs(["Files", "SPCX-style: post-event IV-rich calendars", "NFLX-style: directional diagonals", "Earnings plays (separate)"])

# ------------------------------------------------------------------ files
with tab_data:
    st.subheader("What was loaded")
    nt = pd.DataFrame(notes, columns=["File", "Rows", "Kind", "Type", "Structure"])
    st.dataframe(nt, hide_index=True, use_container_width=True)
    for f, n, *_ in notes:
        if n == 5000:
            st.warning(f"{f}: exactly 5,000 rows - Barchart likely truncated the export, so some symbols may be missing.")
    st.caption("Barchart quirks handled: the short-put diagonal file's 'Net Credit' is really a debit (bear debit diagonal); "
               "long-put diagonal and short-call diagonal are credit structures (kept for reference, not screened).")
    st.dataframe(df.groupby(["Kind", "CP", "Struct"]).size().rename("rows").reset_index(), hide_index=True)

# ------------------------------------------------------------------ SPCX-style
with tab_spcx:
    st.markdown("**Template:** the SPCX trade - short a near-expiry ATM leg priced far above the back month, after the event, "
                "long the next monthly. Pre-earnings names are excluded here (see the earnings tab).")
    a, b, c, d = st.columns(4)
    min_skew = a.slider("Min front-vs-back IV gap (pts)", 2, 25, 8, key="sk")
    max_mny = b.slider("Max |strike vs spot| (%)", 0.5, 6.0, 3.0, 0.5, key="mn")
    max_ds = c.slider("Max short-leg DTE", 1, 21, 7, key="ds")
    min_deb = d.number_input("Min debit ($/share)", 0.0, 50.0, 0.30, key="md")
    e, f = st.columns(2)
    only_post = e.checkbox("Only names already reported", False, key="op")
    sort_by = f.selectbox("Sort by", ["EV_calm%", "EV_base%", "Skew"], key="so1")
    res = sc.spcx_style(df, min_skew, max_mny, max_ds, min_deb)
    res = res[~res.EventFront]
    if file_events:
        st.caption("Names reporting within the next 7 trading days (per your earnings file) are excluded. "
                   "A report later than that can still fall inside a 10/30+ short leg only if the file's window covers it.")
    else:
        st.caption("No earnings file loaded: only the fallback dates are excluded - check any name for an unlisted report.")
    if only_post:
        res = res[res.Reported]
    show(res, sort_by, key="spcx")

# ------------------------------------------------------------------ NFLX-style
with tab_nflx:
    st.markdown("**Template:** the NFLX trade - price bouncing off a put wall / support; buy a later-dated call (or put) "
                "diagonal against a near-dated short beyond the level. Needs cheap long-leg vol and a modest debit.")
    a, b, c, d = st.columns(4)
    direction = a.radio("Direction", ["Bullish", "Bearish"], horizontal=True, key="dir")
    max_ivr = b.slider("Max IV rank", 5, 100, 40, key="ivr")
    max_ivhv = c.slider("Max IV/HV", 0.5, 2.0, 1.05, 0.05, key="ivhv")
    min_delta = d.slider("Min |net delta|", 0.05, 0.6, 0.15, 0.05, key="dl")
    e, f, g, h = st.columns(4)
    max_pct = e.slider("Max debit (% of spot)", 1.0, 20.0, 8.0, 0.5, key="pct")
    max_ds2 = f.slider("Max short DTE", 3, 30, 14, key="ds2")
    drift = g.slider("Expected drift to short expiry (%)", -10.0, 10.0, 2.0 if direction == "Bullish" else -2.0, 0.5, key="drf")
    syms = sorted(df.Symbol.unique())
    pick = h.multiselect("Symbols (blank = all)", syms, key="sy")
    lc, lt = st.columns(2)
    level = lc.number_input("Optional level for the short strike (call wall / target), 0 = off", 0.0, 100000.0, 0.0, key="lv")
    tol = lt.slider("Short strike within (%) of level", 0.5, 10.0, 3.0, 0.5, key="lt")
    res = sc.nflx_style(df, direction, max_ivr, max_ivhv, min_delta, max_pct, max_ds2, level=level or None, level_tol=tol)
    if pick:
        res = res[res.Symbol.isin(pick)]
    res = res[~res.EventFront]
    sort2 = st.selectbox("Sort by", ["EV_base%", "EV_calm%", "Skew"], key="so2")
    st.caption("EV here includes the drift you set (a bounce thesis); set it to 0 to see the no-view value.")
    show(res, sort2, drift=drift, key="nflx")

# ------------------------------------------------------------------ earnings
with tab_earn:
    st.markdown("**Earnings plays are tracked separately.** Calendars here hold the print inside the short leg. "
                "The model splits front/back IV into baseline vol plus a one-off event move, then values the back leg "
                "after the crush. At a fair implied move the expected value is usually negative; the 'calm' column "
                "shows what you need (event move at 70% of implied by default).")
    a, b, c = st.columns(3)
    mn = a.slider("Max |strike vs spot| (%)", 0.5, 6.0, 2.0, 0.5, key="e1")
    mdl = b.number_input("Min debit ($/share)", 0.0, 100.0, 0.5, key="e2")
    srt = c.selectbox("Sort by", ["EV_calm%", "EV_base%", "Skew"], key="so3")
    res = df[(df.EventFront) & (df.Struct == "debit") & (df.Kind == "Calendar") & (df.Mny.abs() <= mn) & (df.Net >= mdl)]
    show(res, srt, key="earn")
