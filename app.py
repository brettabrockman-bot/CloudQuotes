
import os
import sqlite3
from datetime import datetime, date
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Jenne Cloud Quotes Dashboard — Team V1 (Locked Mapping)", layout="wide")
st.title("☁️ Jenne Cloud Quotes Dashboard — Team V1 (Locked Mapping)")

# ------------------------------
# Auth / Config
# ------------------------------
DASH_PASSWORD = os.getenv("DASH_PASSWORD") or st.secrets.get("DASH_PASSWORD", None)
if DASH_PASSWORD:
    pw = st.sidebar.text_input("Enter dashboard password", type="password")
    if pw != DASH_PASSWORD:
        st.stop()

DB_PATH = os.getenv("DB_PATH", "data/quotes.db")

# Regions map (adjust if needed)
REGION_BY_STATE = {
    # South Central
    "TX":"South Central","LA":"South Central","AR":"South Central","MO":"South Central","NM":"South Central","OK":"South Central",
    # West
    "CA":"West","WA":"West","OR":"West","AZ":"West","NV":"West","UT":"West","CO":"West","ID":"West","MT":"West","WY":"West","AK":"West","HI":"West",
    # Midwest
    "IL":"Midwest","IN":"Midwest","IA":"Midwest","KS":"Midwest","MI":"Midwest","MN":"Midwest","NE":"Midwest","ND":"Midwest","OH":"Midwest","SD":"Midwest","WI":"Midwest",
    # Northeast
    "CT":"Northeast","MA":"Northeast","ME":"Northeast","NH":"Northeast","NJ":"Northeast","NY":"Northeast","PA":"Northeast","RI":"Northeast","VT":"Northeast",
    # Southeast
    "AL":"Southeast","FL":"Southeast","GA":"Southeast","KY":"Southeast","MS":"Southeast","NC":"Southeast","SC":"Southeast","TN":"Southeast","VA":"Southeast","WV":"Southeast","DC":"Southeast",
}

ALL_REGIONS = ["All Regions","South Central","West","Midwest","Northeast","Southeast"]

# ------------------------------
# Column mapping locked to provided headers (Row 2 = header index 1)
# ------------------------------
# Exact headers from user (Row 2 in sheet)
RAW_HEADERS = [
    "Account Number","Partner Name","Quote Number","Vendor Name","Part Number","Part Description","Quantity","UnitCost",
    "MonthlyRecurringCharge","Entered Date","Entered By","Cloud Inside Rep","Last Changed Date","Expire Date","Expected Close",
    "Inside Rep.","Territory","State","Summary","Milestone","Grand Total after Discount","Note"
]

# Canonical -> source header mapping
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
    # extras (kept in DB for future use)
    # "expected_close": "Expected Close",
    # "cloud_inside_rep": "Cloud Inside Rep",
    # "territory": "Territory",
    # "mrc": "MonthlyRecurringCharge",
}

EXTRA_FIELDS = {
    "expected_close": "Expected Close",
    "cloud_inside_rep": "Cloud Inside Rep",
    "territory": "Territory",
    "monthly_recurring": "MonthlyRecurringCharge",
    "entered_by": "Entered By",
    "last_changed": "Last Changed Date",
    "expire_date": "Expire Date",
    "summary": "Summary",
    "note": "Note",
    "part_number": "Part Number",
    "quantity": "Quantity",
    "unit_cost": "UnitCost",
}

def parse_stage_bucket(x: str) -> str:
    """Map Milestone to Open/Won/Lost buckets for quick filtering."""
    s = str(x or "").strip().lower()
    if any(k in s for k in ["won","booked","closed-won"]):
        return "Won"
    if any(k in s for k in ["lost","closed-lost","declined","cancelled","canceled"]):
        return "Lost"
    return "Open"

# ------------------------------
# DB Helpers
# ------------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
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
                source_file TEXT,
                -- extras
                expected_close TEXT,
                cloud_inside_rep TEXT,
                territory TEXT,
                monthly_recurring REAL,
                entered_by TEXT,
                last_changed TEXT,
                expire_date TEXT,
                summary TEXT,
                note TEXT
            );
            """
        )
        conn.commit()

def add_region(df: pd.DataFrame) -> pd.DataFrame:
    st_col = df.get("state")
    if st_col is None:
        df["region"] = None
        return df
    tmp = st_col.astype(str).str.upper().str.strip().str.extract(r'([A-Z]{2})')[0]
    df["state"] = tmp
    df["region"] = tmp.map(REGION_BY_STATE).fillna("Unassigned")
    return df

def read_uploaded(uploaded_file):
    name = uploaded_file.name.lower()
    # Header row is Row 2 => header=1 (0-index) for Excel
    if name.endswith(".csv"):
        # For CSV, assume first row is header; if this export produces a dummy first row, we can skiprows=1
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, header=1)  # <-- row 2 headers
    return df

def upsert_quotes_from_df(raw_df: pd.DataFrame, source_file: str, week_label: str):
    df = raw_df.copy()

    # Ensure required headers exist
    missing = [LOCKED_MAP[k] for k in ["quote_id","amount"] if LOCKED_MAP[k] not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}. Please verify the export uses the standard layout.")
        st.stop()

    # Rename to canonical
    rename_pairs = {src: tgt for tgt, src in LOCKED_MAP.items() if src in df.columns}
    df = df.rename(columns=rename_pairs)

    # Keep extras with canonical names
    for canon, src in EXTRA_FIELDS.items():
        if src in df.columns:
            df[canon] = df[src]

    # Types
    df['quote_id'] = df['quote_id'].astype(str).str.strip()
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

    if 'quote_date' in df.columns:
        df['quote_date'] = pd.to_datetime(df['quote_date'], errors='coerce').dt.date

    if 'state' in df.columns:
        df['state'] = df['state'].astype(str).str.upper().str.strip()
        df['state'] = df['state'].str.extract(r'([A-Z]{2})')

    # Stage bucket from Milestone
    df['stage_bucket'] = df['stage'].apply(parse_stage_bucket) if 'stage' in df.columns else None

    # Region
    df = add_region(df)

    # Add metadata
    df['source_file'] = source_file
    df['week_label'] = week_label

    # Write
    with sqlite3.connect(DB_PATH) as conn:
        cols = ['quote_id','account','amount','stage','stage_bucket','owner','vendor','product_family','state','quote_date',
                'week_label','source_file','expected_close','cloud_inside_rep','territory','monthly_recurring','entered_by',
                'last_changed','expire_date','summary','note']
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df[cols].to_sql('quotes', conn, if_exists='append', index=False)

def load_quotes():
    with sqlite3.connect(DB_PATH) as conn:
        q = "SELECT * FROM quotes"
        df = pd.read_sql(q, conn, parse_dates=['quote_date'])
    df = add_region(df)
    return df

init_db()

# ------------------------------
# Sidebar: Upload + Filters
# ------------------------------
st.sidebar.header("Weekly Upload")
uploaded = st.sidebar.file_uploader("Upload weekly Excel/CSV export", type=["xlsx","xls","csv"])
week_label = st.sidebar.text_input("Week label (e.g., 2025-W45)", value=datetime.now().strftime("%Y-W%W"))
run_ingest = st.sidebar.button("Ingest File")

if run_ingest and uploaded is not None:
    try:
        raw = read_uploaded(uploaded)
        upsert_quotes_from_df(raw, source_file=uploaded.name, week_label=week_label)
        st.sidebar.success(f"Ingested: {uploaded.name}")
    except Exception as e:
        st.sidebar.error(f"Ingest failed: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
region_choice = st.sidebar.selectbox("Region", ALL_REGIONS, index=0)
date_min = st.sidebar.date_input("Start date", value=date(2025,1,1))
date_max = st.sidebar.date_input("End date", value=date.today())
owner_filter = st.sidebar.text_input("Owner contains")
vendor_filter = st.sidebar.text_input("Vendor contains")
account_filter = st.sidebar.text_input("Account contains")
stage_options = ["All","Open","Won","Lost"]
stage_choice = st.sidebar.selectbox("Stage (approx)", stage_options, index=0)

# ------------------------------
# Data Load + Global Filters
# ------------------------------
df = load_quotes()
if df.empty:
    st.info("No data yet. Upload your first weekly export on the left to get started.")
    st.stop()

df['quote_date'] = pd.to_datetime(df['quote_date']).dt.date
df = df[(df['quote_date'] >= date_min) & (df['quote_date'] <= date_max)]

if region_choice != "All Regions":
    df = df[df['region'] == region_choice]

if owner_filter:
    df = df[df['owner'].astype(str).str.contains(owner_filter, case=False, na=False)]
if vendor_filter:
    df = df[df['vendor'].astype(str).str.contains(vendor_filter, case=False, na=False)]
if account_filter:
    df = df[df['account'].astype(str).str.contains(account_filter, case=False, na=False)]

if stage_choice != "All":
    df = df[df['stage_bucket'] == stage_choice]

# ------------------------------
# KPIs
# ------------------------------
total_quotes = len(df)
total_amount = df['amount'].sum()
unique_accounts = df['account'].nunique(dropna=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Quotes", f"{total_quotes:,}")
col2.metric("Total Amount", f"${total_amount:,.0f}")
col3.metric("Unique Accounts", f"{unique_accounts:,}")
col4.metric("Region", region_choice)

# ------------------------------
# Charts (clickable)
# ------------------------------
st.markdown("### Trends & Breakdowns (click a point/bar to drill down)")
df['week'] = df['quote_date'].apply(lambda d: f"{d.isocalendar()[0]}-W{int(d.isocalendar()[1]):02d}" if pd.notnull(d) else None)

weekly = df.groupby('week', as_index=False)['amount'].sum().sort_values('week')
fig_week = px.line(weekly, x='week', y='amount', markers=True, title="Weekly Quote Amount")
clicked_week = plotly_events(fig_week, click_event=True, hover_event=False, select_event=False, override_height=400, key="week_click")

top_accts = df.groupby('account', dropna=True, as_index=False)['amount'].sum().sort_values('amount', ascending=False).head(15)
fig_acct = px.bar(top_accts, x='account', y='amount', title="Top Accounts")
clicked_acct = plotly_events(fig_acct, click_event=True, hover_event=False, select_event=False, override_height=420, key="acct_click")

top_vendors = df.groupby('vendor', dropna=True, as_index=False)['amount'].sum().sort_values('amount', ascending=False).head(15)
fig_vendor = px.bar(top_vendors, x='vendor', y='amount', title="Top Vendors")
clicked_vendor = plotly_events(fig_vendor, click_event=True, hover_event=False, select_event=False, override_height=420, key="vendor_click")

by_state = df.groupby('state', dropna=True, as_index=False)['amount'].sum().sort_values('amount', ascending=False)
fig_state = px.bar(by_state, x='state', y='amount', title="By State")
clicked_state = plotly_events(fig_state, click_event=True, hover_event=False, select_event=False, override_height=400, key="state_click")

# ------------------------------
# Drilldown
# ------------------------------
drill_label = None
drilled = None

if clicked_week:
    wk = clicked_week[0].get('x')
    drill_label = f"Week = {wk}"
    drilled = df[df['week'] == wk]
elif clicked_acct:
    acct = clicked_acct[0].get('x')
    drill_label = f"Account = {acct}"
    drilled = df[df['account'] == acct]
elif clicked_vendor:
    vend = clicked_vendor[0].get('x')
    drill_label = f"Vendor = {vend}"
    drilled = df[df['vendor'] == vend]
elif clicked_state:
    stt = clicked_state[0].get('x')
    drill_label = f"State = {stt}"
    drilled = df[df['state'] == stt]

st.markdown("---")
if drilled is not None and not drilled.empty:
    st.subheader(f"Drilldown: {drill_label}")
    k1,k2,k3 = st.columns(3)
    k1.metric("Quotes", f"{len(drilled):,}")
    k2.metric("Amount", f"${drilled['amount'].sum():,.0f}")
    k3.metric("Unique Accounts", f"{drilled['account'].nunique(dropna=True):,}")
    st.dataframe(drilled.sort_values('quote_date', ascending=False), use_container_width=True)
    st.download_button("Download drilldown CSV", drilled.to_csv(index=False).encode('utf-8'), file_name="drilldown.csv", mime="text/csv")
else:
    st.subheader("Quotes Table")
    st.dataframe(df.sort_values('quote_date', ascending=False), use_container_width=True)
    st.download_button("Download filtered CSV", df.to_csv(index=False).encode('utf-8'), file_name="filtered_quotes.csv", mime="text/csv")

# ------------------------------
# Region Tabs (Team View)
# ------------------------------
st.markdown("---")
st.header("Team View by Region")
tabs = st.tabs([r for r in ALL_REGIONS if r != "All Regions"])

def region_block(region_name: str):
    rdf = df if region_name == "All Regions" else df[df['region'] == region_name]
    if rdf.empty:
        st.info(f"No data for {region_name}.")
        return
    c1,c2,c3 = st.columns(3)
    c1.metric("Quotes", f"{len(rdf):,}")
    c2.metric("Amount", f"${rdf['amount'].sum():,.0f}")
    c3.metric("Accounts", f"{rdf['account'].nunique(dropna=True):,}")
    by_owner = rdf.groupby('owner', as_index=False)['amount'].sum().sort_values('amount', ascending=False).head(15)
    fig = px.bar(by_owner, x='owner', y='amount', title=f"{region_name}: Top Owners")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rdf.sort_values('quote_date', ascending=False), use_container_width=True)

for t, region_name in zip(tabs, [r for r in ALL_REGIONS if r != "All Regions"]):
    with t:
        region_block(region_name)

st.caption("Team V1 — Locked to Epicor Row 2 headers; regions + clickable drilldowns.")

# Mapping preview (optional)
with st.expander("Show locked column mapping"):
    st.write(pd.DataFrame({
        "Canonical Field": list(LOCKED_MAP.keys()),
        "Source Header": [LOCKED_MAP[k] for k in LOCKED_MAP.keys()]
    }))
