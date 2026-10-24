"""Core logic for the Barchart calendar/diagonal play screener.

Leg convention after loading (every row, every file type):
  long  leg = the leg we BUY  (price = ask)
  short leg = the leg we SELL (price = bid)
  net   = long price - short price  (>0 debit, <0 credit)
"""
import re, glob, os
from math import sqrt
import numpy as np
import pandas as pd
from scipy.stats import norm

R = 0.04
ETFS = {"SPY","QQQ","IWM","DIA","XLF","XLE","XLK","XLV","XLY","XLP","XLI","XLU","SMH","SOXX","TLT","GLD","SLV","USO","EEM","EFA","HYG","ARKK","VXX","UVXY","SQQQ","TQQQ","SPXL","SPXS"}
EXIT_HAIRCUT = 0.97          # sell the long leg ~3% under model mid

DEFAULT_EVENTS = """GOOGL 2026-10-27 AMC
GOOG 2026-10-27 AMC
KO 2026-10-27 BMO
MSFT 2026-10-28 AMC
GEV 2026-10-28 BMO
AMZN 2026-10-29 AMC
INTC 2026-10-29 AMC
CVX 2026-10-30 BMO"""

DEFAULT_RECENT = """SPCX 2026-10-20 AMC
META 2026-10-21 AMC
IBM 2026-10-21 AMC
TSLA 2026-10-21 AMC"""


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", "").str.strip(), errors="coerce")


def file_date(name):
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", os.path.basename(str(name)))
    return pd.Timestamp(int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else None


def load_barchart(file, asof, label=""):
    """Normalise any of the six Barchart screener exports (column-based detection)."""
    df = pd.read_csv(file)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df[pd.to_numeric(df["Price~"].astype(str).str.replace(",", ""), errors="coerce").notna()].copy()
    cols = set(df.columns)
    is_cal = "Leg2 Strike" not in cols
    px_a = "Bid1" if "Bid1" in cols else "Ask1"
    px_b = "Bid2" if "Bid2" in cols else "Ask2"
    a_is_long = px_a == "Ask1"                       # diagonal debit files: Leg1 is bought
    cp = df["Type"].astype(str).str.strip().str[0].str.upper()
    ka = _num(df["Leg1 Strike"])
    kb = ka if is_cal else _num(df["Leg2 Strike"])
    A = dict(exp=pd.to_datetime(df["Exp Leg1"]), k=ka, px=_num(df[px_a]), iv=_num(df["Leg1 IV"]))
    B = dict(exp=pd.to_datetime(df["Exp Leg2"]), k=kb, px=_num(df[px_b]), iv=_num(df["Leg2 IV"]))
    L, S = (A, B) if a_is_long else (B, A)
    out = pd.DataFrame({
        "Symbol": df["Symbol"].astype(str).str.strip(), "Px": _num(df["Price~"]), "CP": cp,
        "Kind": "Calendar" if is_cal else "Diagonal",
        "ExpS": S["exp"], "ExpL": L["exp"], "KS": S["k"], "KL": L["k"],
        "PxS": S["px"], "PxL": L["px"], "IVS": S["iv"], "IVL": L["iv"],
        "IVR": _num(df["IV Rank"]), "IVHV": _num(df["IV/HV"]),
        "Delta": _num(df["Net Delta"]), "Vega": _num(df["Net Vega"]),
    })
    out["Net"] = out.PxL - out.PxS
    out["Struct"] = np.where(out.Net > 0, "debit", "credit")
    out["DTE_S"] = (out.ExpS - asof).dt.days
    out["DTE_L"] = (out.ExpL - asof).dt.days
    out["Mny"] = (out.KS / out.Px - 1) * 100
    out["Skew"] = out.IVS - out.IVL                 # + => short leg richer
    out["File"] = label
    out = out[(out.ExpS < out.ExpL) & (out.DTE_S >= 0)].reset_index(drop=True)
    return out


def parse_events(text, asof=None):
    ev = {}
    for line in text.splitlines():
        p = line.replace(",", " ").split()
        if len(p) >= 2:
            try:
                d = pd.Timestamp(p[1])
            except Exception:
                continue
            t = p[2].upper() if len(p) > 2 else "AMC"
            ev[p[0].upper()] = (d, t)
    return ev


def is_earnings_file(src):
    try:
        return "Earnings Date" in [x.strip() for x in pd.read_csv(src, nrows=0).columns]
    except Exception:
        return False


def parse_earnings_file(src):
    """Barchart 'earnings in next 7 trading days' export -> {SYMBOL: (date, 'AMC'|'BMO')}.
    The 'Released' column holds the timing; unknown ('--') is treated as BMO (earlier reaction = conservative)."""
    d = pd.read_csv(src)
    d.columns = [x.strip() for x in d.columns]
    d = d[pd.to_datetime(d["Earnings Date"], errors="coerce").notna()]
    out = {}
    for sym, dt, t in zip(d["Symbol"].astype(str).str.strip().str.upper(), pd.to_datetime(d["Earnings Date"]), d["Released"].astype(str)):
        out[sym] = (dt, "AMC" if "after" in t.lower() else "BMO")
    return out


def tag_events(df, events, recent):
    """EventFront: print lands before the short leg expires. EventBack: only the long leg holds it."""
    ef, eb, ed, rec = [], [], [], []
    for sym, es, el in zip(df.Symbol, df.ExpS, df.ExpL):
        e = events.get(sym.upper())
        if e:
            react = e[0] + pd.Timedelta(days=1 if e[1] == "AMC" else 0)
            ef.append(react <= es); eb.append(es < react <= el); ed.append(e[0])
        else:
            ef.append(False); eb.append(False); ed.append(pd.NaT)
        rec.append(sym.upper() in recent)
    df = df.copy()
    df["EventFront"], df["EventBack"], df["EventDate"], df["Reported"] = ef, eb, ed, rec
    return df


# ------------------------------------------------------------------ pricing / EV model
def bs(S, K, T, v, cp):
    T = max(T, 1e-6)
    d1 = (np.log(S / K) + (R + .5 * v * v) * T) / (v * np.sqrt(T)); d2 = d1 - v * np.sqrt(T)
    if cp == "C":
        return S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
    return K * np.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _intrinsic(S, K, cp):
    return np.maximum(S - K, 0) if cp == "C" else np.maximum(K - S, 0)


def _terminal(r, mult, scale, drift, z):
    """Spot distribution + back-leg IV at the short-leg expiry for one row."""
    Ts, Tl, vs, vl = r.DTE_S / 365, r.DTE_L / 365, r.IVS / 100, r.IVL / 100
    if r.EventFront and Tl > Ts:
        sb2 = (vl * vl * Tl - vs * vs * Ts) / (Tl - Ts)       # baseline vol implied by the two tenors
        sb = sqrt(max(sb2, 4e-4)); se2 = max(vs * vs * Ts - sb * sb * Ts, 0.0)
        tot = sqrt(scale ** 2 * se2 + (mult * sb) ** 2 * Ts); back_iv = sb
    else:
        tot = mult * vl * sqrt(Ts); back_iv = vl
    ST = r.Px * np.exp(drift - .5 * tot ** 2 + tot * z)
    return ST, back_iv


def ev_for_row(r, mult=1.0, scale=1.0, drift=0.0, z=None):
    ST, back_iv = _terminal(r, mult, scale, drift, z)
    long_v = bs(ST, r.KL, (r.DTE_L - r.DTE_S) / 365, back_iv, r.CP) * EXIT_HAIRCUT
    pl = long_v - _intrinsic(ST, r.KS, r.CP) - r.Net
    return pl.mean() / r.Net * 100, (pl > 0).mean() * 100


def add_ev(df, calm_mult=0.85, calm_scale=0.7, drift_pct=0.0, n=4000, seed=7):
    z = np.random.default_rng(seed).standard_normal(n)
    drift = np.log1p(drift_pct / 100)
    b, bp, c, cpop = [], [], [], []
    for r in df.itertuples():
        e1, p1 = ev_for_row(r, 1.0, 1.0, drift, z)
        e2, p2 = ev_for_row(r, calm_mult, calm_scale, drift, z)
        b.append(e1); bp.append(p1); c.append(e2); cpop.append(p2)
    df = df.copy()
    df["EV_base%"], df["PoP_base%"], df["EV_calm%"], df["PoP_calm%"] = b, bp, c, cpop
    return df


def payoff_curve(r, mult=1.0, points=61, span=0.18):
    """P/L per share at short expiry, flat IV, no event."""
    S = np.linspace(r.Px * (1 - span), r.Px * (1 + span), points)
    back_iv = r.IVL / 100 * mult
    pl = bs(S, r.KL, (r.DTE_L - r.DTE_S) / 365, back_iv, r.CP) * EXIT_HAIRCUT - _intrinsic(S, r.KS, r.CP) - r.Net
    return pd.DataFrame({"Price": S, "P/L per share": pl}).set_index("Price")


# ------------------------------------------------------------------ the two discovered plays
def spcx_style(df, min_skew=8, max_abs_mny=3.0, max_dte_s=7, min_debit=0.30, kinds=("Calendar",)):
    """Post-event IV-rich calendar: short front leg priced far above the back, near ATM, expiring soon."""
    m = (df.Struct == "debit") & df.Kind.isin(kinds) & (df.Mny.abs() <= max_abs_mny) & \
        (df.DTE_S <= max_dte_s) & (df.Skew >= min_skew) & (df.Net >= min_debit)
    return df[m]


def nflx_style(df, direction="Bullish", max_ivr=40, max_ivhv=1.05, min_abs_delta=0.15,
               max_debit_pct=8.0, max_dte_s=14, min_gap=10, level=None, level_tol=3.0, min_dte_l=14):
    """Directional diagonal into a bounce / break: long later-dated leg, short near leg beyond the level."""
    cp = "C" if direction == "Bullish" else "P"
    m = (df.Struct == "debit") & (df.Kind == "Diagonal") & (df.CP == cp) & (df.Delta.abs() >= min_abs_delta) & \
        (np.sign(df.Delta) == (1 if cp == "C" else -1)) & (df.IVR <= max_ivr) & (df.IVHV <= max_ivhv) & \
        (df.Net / df.Px * 100 <= max_debit_pct) & (df.DTE_S <= max_dte_s) & \
        ((df.DTE_L - df.DTE_S) >= min_gap) & (df.DTE_L >= min_dte_l)
    if level:
        m &= ((df.KS / level - 1).abs() * 100 <= level_tol)
    return df[m]
