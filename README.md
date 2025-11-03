
# Jenne Cloud Quotes Dashboard — Team V1 (Deduped + Upsert Fix)

- Handles duplicate Quote Numbers gracefully.
- Collapses multiple line items to one quote-level row.
- Replaces existing quote_ids on re-ingest (no UNIQUE constraint errors).
- Default header row = 3 (adjust in sidebar).

## How to Deploy
1. Upload files to your GitHub repo.
2. Main file path = `app.py`.
3. Python = 3.11 (set in Streamlit Cloud Advanced settings).

## Quick Start
- Upload the Epicor export as-is.
- If you re-upload the same file, existing quotes will be replaced.
