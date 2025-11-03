
# Jenne Cloud Quotes Dashboard — Team V1 (Full, with Diagnostics)

**What’s included**
- Header row control (default 3) for Epicor exports
- Strict or flexible header mapping (with aliases)
- Deduplication to quote-level + upsert (no UNIQUE constraint errors)
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
- Upload, set week label, ingest

## Troubleshooting
- If data looks light, open **Data quality & diagnostics** to see row/quote counts and min/max dates.
- If totals look off, switch **Quote total aggregation** between `sum` and `max` depending on whether your report totals repeat per line.
