
# Jenne Cloud Quotes Dashboard — Team V1 (Row-3 Headers & Drilldowns)

Deployed on Streamlit Community Cloud. Supports `.xls/.xlsx/.csv`. Region tabs + clickable drilldowns.

## Key Behaviors
- **Header row is configurable** in the sidebar (default **3** for your Epicor export).
- **Strict mapping** (Epicor exact headers) or **Flexible mapping** (aliases & normalization).
- Stores history in SQLite (`data/quotes.db`).
- KPIs + weekly trend + top accounts/vendors + by-state + drilldown table.

## How to Run (Streamlit Cloud)
1. Ensure **Python 3.11** is selected in **Advanced settings** at deploy time.
2. Main file path = `app.py`.
3. (Optional) Secrets:
   - `DASH_PASSWORD` for a simple gate.
   - `DB_PATH` (default `data/quotes.db`).

## Weekly Ingest
1. Export the Epicor “Cloud Quotes” report (unaltered).
2. In the app, set **Header row (1-based)** to **3** (or whatever line your headers are on).
3. Upload file, set Week label (e.g., `2025-W45`), click **Ingest**.

## Troubleshooting
- **Missing required columns:** Check the **Header row** setting. If headers vary, uncheck **Strict Epicor mapping** to use alias-matching.
- **No data after restart:** SQLite should persist, but if you need stronger durability later, swap to Turso/Supabase (UI unchanged).
