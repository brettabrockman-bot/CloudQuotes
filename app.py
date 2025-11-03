
import os
import re
import sqlite3
from datetime import datetime, date
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events

# ──────────────────────────────────────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Jenne Cloud Quotes Dashboard — Team V1", layout="wide")
st.title("☁️ Jenne Cloud Quotes Dashboard — Team V1")

DASH_PASSWORD = os.getenv("DASH_PASSWORD") or st.secrets.get("DASH_PASSWORD", None)
if DASH_PASSWORD:
    pw = st.sidebar.text_input("Enter dashboard password", type="password")
    if pw != DASH_PASSWORD:
        st.stop()

DB_PATH = os.getenv("DB_PATH", "data/quotes.db")

# ──────────────────────────────────────────────────────────────────────────────
# Region map
# ──────────────────────────────────────────────────────────────────────────────
REGION_BY_STATE = {
    "TX":"South Central","LA":"South Central","AR":"South Central","MO":"South Central","NM":"South Central","OK":"South Central",
    "CA":"West","WA":"West","OR":"West","AZ":"West","NV":"West","UT":"West","CO":"West","ID":"West","MT":"West","WY":"West","AK":"West","HI":"West",
    "IL":"Midwest","IN":"Midwest","IA":"Midwest","KS":"Midwest","MI":"Midwest","MN":"Midwest","NE":"Midwest","ND":"Midwest","OH":"Midwest","SD":"Midwest","WI":"Midwest",
    "CT":"Northeast","MA":"Northeast","ME":"Northeast","NH":"Northeast","NJ":"Northeast","NY":"Northeast","PA":"Northeast","RI":"Northeast","VT":"Northeast",
    "AL":"Southeast","FL":"Southeast","GA":"Southeast","KY":"Southeast","MS":"Southeast","NC":"Southeast","SC":"Southeast","TN":"Southeast","VA":"Southeast","WV":"Southeast","DC":"Southeast",
}
ALL_REGIONS = ["All Regions","South Central","West","Midwest","Northeast","Southeast"]

# ──────────────────────────────────────────────────────────────────────────────
# Header handling & mapping
# ──────────────────────────────────────────────────────────────────────────────
LOCKED_MAP = {
    "account": "Partner Name",
    "quote_id": "Quote Number",
    "vendor": "Vendor Name",
    "amount": "Grand Total after Discount",
    "owner": "Inside Rep.",
    "stage": "Milestone",
    "quote_date": "Entered Date",
    "state": "State",
    "product_family": "Part Description",
}

PREFERRED = {
    "account": "partner name",
    "quote_id": "quote number",
    "vendor": "vendor name",
    "amount": "grand total after discount",
    "owner": "inside rep.",
    "stage": "milestone",
    "quote_date": "entered date",
    "state": "state",
    "product_family": "part description",
}
ALIASES = {
    "quote number": ["quote no", "quote #", "quote id", "quote_num"],
    "grand total after discount": ["grand total", "total after discount", "quote total", "total amount"],
    "inside rep.": ["inside rep", "inside representative", "rep", "owner"],
    "entered date": ["entered", "created date", "created on", "quote date"],
    "part description": ["description", "product description"],
}
REQUIRED_CANONICAL = ["quote_id", "amount"]

def normalize(s):
    s = str(s or "")
    s = s.strip().lower()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"[^a-z0-9 #./()\-]+", "", s)
    return s.strip()

def flex_map_columns(df):
    norm_to_orig = {normalize(c): c for c in df.columns}
    mapped = {}
    for canon, pref in PREFERRED.items():
        candidates = [pref] + ALIASES.get(pref, [])
        for cand in candidates:
            if normalize(cand) in norm_to_orig:
                mapped[canon] = norm_to_orig[normalize(cand)]
                break
    return mapped

def parse_stage_bucket(x: str) -> str:
    s = str(x or "").strip().lower()
    if any(k in s for k in ["won", "booked", "closed-won"]):
        return "Won"
    if any(k in s for k in ["lost", "closed-lost", "declined", "cancelled", "canceled"]):
        return "Lost"
    return "Open"

# ──────────────────────────────────────────────────────────────────────────────
# Database setup
# ──────────────────────────────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                quote_id TEXT PRIMARY KEY,
                account TEXT,
                amount REAL,
                stage TEXT,
                stage_bucket TEXT,
                owner TEXT,
                vendor TEXT,
                product_family TEXT,
                state TEXT,
                quote_date TEXT,
                week_label TEXT,
                source_file TEXT
            );
        """)
        conn.commit()

def add_region(df: pd.DataFrame) -> pd.DataFrame:
    st_col = df.get("state")
    if st_col is None:
        df["region"] = None
        return df
    tmp = st_col.astype(str).str.upper().str.strip().str.extract(r"([A-Z]{2})")[0]
    df["state"] = tmp
    df["region"] = tmp.map(REGION_BY_STATE).fillna("Unassigned")
    return df

def dedupe_to_quote_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse line-item rows to one row per quote_id."""
    df = df.copy()
    sort_cols = []
    if "last_changed" in df.columns:
        df["last_changed"] = pd.to_datetime(df["last_changed"], errors="coerce")
        sort_cols.append("last_changed")
    if "quote_date" in df.columns:
        df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")
        sort_cols.append("quote_date")
    if sort_cols:
        df = df.sort_values(sort_cols)

    agg_map = {
        "account": "last",
        "amount": "max",
        "stage": "last",
        "stage_bucket": "last",
        "owner": "last",
        "vendor": "last",
        "product_family": "last",
        "state": "last",
        "quote_date": "last",
        "week_label": "last",
        "source_file": "last",
        "region": "last",
    }
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}
    deduped = df.groupby("quote_id", as_index=False).agg(agg_map)
    return deduped

# ──────────────────────────────────────────────────────────────────────────────
# Upload & ingest
# ──────────────────────────────────────────────────────────────────────────────
def read_with_header_row(uploaded_file, header_row_1based: int):
    hdr = max(0, int(header_row_1based) - 1)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, skiprows=hdr, header=0)
    else:
        df = pd.read_excel(uploaded_file, header=hdr)
    return df

def upsert_quotes_from_df(raw_df, source_file, week_label, use_locked_map, replace_existing):
    df = raw_df.copy()

    if use_locked_map:
        rename_pairs = {src: tgt for tgt, src in LOCKED_MAP.items() if src in df.columns}
        df = df.rename(columns=rename_pairs)
    else:
        colmap = flex_map_columns(df)
        rename_pairs = {src: tgt for tgt, src in colmap.items()}
        df = df.rename(columns=rename_pairs)

    for req in REQUIRED_CANONICAL:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")

    df["quote_id"] = df["quote_id"].astype(str).str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    if "quote_date" in df.columns:
        df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce").dt.date
    else:
        df["quote_date"] = date.today()

    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.upper().str.strip()
        df["state"] = df["state"].str.extract(r"([A-Z]{2})")

    if "stage" in df.columns:
        df["stage_bucket"] = df["stage"].apply(parse_stage_bucket)
    else:
        df["stage_bucket"] = "Open"

    df = add_region(df)
    df["source_file"] = source_file
    df["week_label"] = week_label
    df = dedupe_to_quote_level(df)

    with sqlite3.connect(DB_PATH) as conn:
        cols = ["quote_id","account","amount","stage","stage_bucket","owner","vendor","product_family","state","quote_date","week_label","source_file"]
        for c in cols:
            if c not in df.columns:
                df[c] = None

        if replace_existing:
            ids = df["quote_id"].dropna().astype(str).unique().tolist()
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM quotes WHERE quote_id IN ({placeholders})", ids)

        df[cols].to_sql("quotes", conn, if_exists="append", index=False)

def load_quotes():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM quotes", conn, parse_dates=["quote_date"])
    return add_region(df)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar UI
# ──────────────────────────────────────────────────────────────────────────────
init_db()
st.sidebar.header("Weekly Upload")
uploaded = st.sidebar.file_uploader("Upload Epicor export (.xls/.xlsx/.csv)", type=["xlsx","xls","csv"])

header_row_1based = st.sidebar.number_input("Header row (1-based)", min_value=1, max_value=20, value=3, step=1)
use_locked_map = st.sidebar.checkbox("Strict Epicor mapping (exact header names)", value=True)
replace_existing = st.sidebar.checkbox("Replace existing Quote Numbers on ingest", value=True)

week_label = st.sidebar.text_input("Week label (e.g., 2025-W45)", value=datetime.now().strftime("%Y-W%W"))
if st.sidebar.button("Ingest File") and uploaded is not None:
    try:
        raw = read_with_header_row(uploaded, header_row_1based)
        upsert_quotes_from_df(raw, uploaded.name, week_label, use_locked_map, replace_existing)
        st.sidebar.success(f"Ingested: {uploaded.name}")
    except Exception as e:
        st.sidebar.error(f"Ingest failed: {e}")

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
region_choice = st.sidebar.selectbox("Region", ALL_REGIONS, index=0)
date_min = st.sidebar.date_input("Start date", value=date(2025,1,1))
date_max = st.sidebar.date_input("End date", value=date.today())
owner_filter = st.sidebar.text_input("Owner contains")
vendor_filter = st.sidebar.text_input("Vendor contains")
account_filter = st.sidebar.text_input("Account contains")
stage_choice = st.sidebar.selectbox("Stage (approx)", ["All","Open","Won","Lost"], index=0)

# ──────────────────────────────────────────────────────────────────────────────
# Data + Filters
# ──────────────────────────────────────────────────────────────────────────────
df = load_quotes()
if df.empty:
    st.info("No data yet. Upload a weekly export to get started.")
    st.stop()

df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.date
df = df[(df["quote_date"] >= date_min) & (df["quote_date"] <= date_max)]

if region_choice != "All Regions":
    df = df[df["region"] == region_choice]
if owner_filter:
    df = df[df["owner"].astype(str).str.contains(owner_filter, case=False, na=False)]
if vendor_filter:
    df = df[df["vendor"].astype(str).str.contains(vendor_filter, case=False, na=False)]
if account_filter:
    df = df[df["account"].astype(str).str.contains(account_filter, case=False, na=False)]
if stage_choice != "All":
    df = df[df["stage_bucket"] == stage_choice]

# ──────────────────────────────────────────────────────────────────────────────
# KPIs
# ──────────────────────────────────────────────────────────────────────────────
total_quotes = len(df)
total_amount = df["amount"].sum()
unique_accounts = df["account"].nunique(dropna=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Quotes", f"{total_quotes:,}")
c2.metric("Total Amount", f"${total_amount:,.0f}")
c3.metric("Unique Accounts", f"{unique_accounts:,}")
c4.metric("Region", region_choice)

# ──────────────────────────────────────────────────────────────────────────────
# Charts + Drilldowns
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Trends & Breakdowns (click to drill down)")
df["week"] = df["quote_date"].apply(lambda d: f"{d.isocalendar()[0]}-W{int(d.isocalendar()[1]):02d}" if pd.notnull(d) else None)

weekly = df.groupby("week", as_index=False)["amount"].sum().sort_values("week")
fig_week = px.line(weekly, x="week", y="amount", markers=True, title="Weekly Quote Amount")
clicked_week = plotly_events(fig_week, click_event=True, hover_event=False, select_event=False, key="week_click")

top_accts = df.groupby("account", dropna=True, as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(15)
fig_acct = px.bar(top_accts, x="account", y="amount", title="Top Accounts")
clicked_acct = plotly_events(fig_acct, click_event=True, hover_event=False, select_event=False, key="acct_click")

top_vendors = df.groupby("vendor", dropna=True, as_index=False)["amount"].sum().sort_values("amount", ascending=False).head(15)
fig_vendor = px.bar(top_vendors, x="vendor", y="amount", title="Top Vendors")
clicked_vendor = plotly_events(fig_vendor, click_event=True, hover_event=False, select_event=False, key="vendor_click")

by_state = df.groupby("state", dropna=True, as_index=False)["amount"].sum().sort_values("amount", ascending=False)
fig_state = px.bar(by_state, x="state", y="amount", title="By State")
clicked_state = plotly_events(fig_state, click_event=True, hover_event=False, select_event=False, key="state_click")

# Drilldown logic
drilled, drill_label = None, None
if clicked_week:
    wk = clicked_week[0].get("x"); drill_label = f"Week = {wk}"; drilled = df[df["week"] == wk]
elif clicked_acct:
    acct = clicked_acct[0].get("x"); drill_label = f"Account = {acct}"; drilled = df[df["account"] == acct]
elif clicked_vendor:
    vend = clicked_vendor[0].get("x"); drill_label = f"Vendor = {vend}"; drilled = df[df["vendor"] == vend]
elif clicked_state:
    stt = clicked_state[0].get("x"); drill_label = f"State = {stt}"; drilled = df[df["state"] == stt]

st.markdown("---")
if drilled is not None and not drilled.empty:
    st.subheader(f"Drilldown: {drill_label}")
    st.dataframe(drilled.sort_values("quote_date", ascending=False), use_container_width=True)
    st.download_button("Download drilldown CSV", drilled.to_csv(index=False).encode("utf-8"), "drilldown.csv")
else:
    st.subheader("Quotes Table")
    st.dataframe(df.sort_values("quote_date", ascending=False), use_container_width=True)
    st.download_button("Download filtered CSV", df.to_csv(index=False).encode("utf-8"), "filtered_quotes.csv")
