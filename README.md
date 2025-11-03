
# Jenne Cloud Quotes Dashboard — Team V1 (Combine Line Items)

**What’s included**
- Header row control (default 3) for Epicor exports
- Strict or flexible header mapping (with aliases)
- Combine multiple line items into one quote (SUM or MAX amount)
- MRC support (sums to mrc_total)
- Upsert on re-ingest (no UNIQUE constraint errors)
- Diagnostics sidebar + data quality expander
- Region filters, KPIs, weekly trend, top accounts/vendors, by-state, drilldowns

## Deploy on Streamlit Community Cloud
1) Create app from this repo; **Main file path:** `app.py`
2) **Advanced settings → Python version:** 3.11
3) (Optional) Secrets:
   - `DASH_PASSWORD` to gate access
   - `DB_PATH` (default: `data/quotes.db`)

## Weekly workflow
- Export Epicor report unchanged
- Set **Header row (1-based)** to the line with real headers (usually 3)
- Choose **Quote total aggregation** = `sum` (recommended) or `max`
- Upload, set week label, ingest
