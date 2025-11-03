
# Jenne Cloud Quotes Dashboard — Team V1 (Locked Mapping)

This build is pinned to your exact Epicor export headers (Row 2 is the header row). It supports `.xlsx`, `.xls`, and `.csv`.

## Deploy on Streamlit Community Cloud (Private)
1. Push these files to a GitHub repo.
2. In Streamlit Cloud, create an app from that repo.
3. (Optional) Add Secrets:
   - `DASH_PASSWORD`: simple password gate
   - `DB_PATH`: default `data/quotes.db`

## Upload Instructions
- Upload the weekly export **exactly as Epicor produces it**. The app reads **Row 2** as headers.
- The following fields are mapped:
  - account ← Partner Name
  - quote_id ← Quote Number
  - vendor ← Vendor Name
  - amount ← Grand Total after Discount
  - owner ← Inside Rep.
  - stage ← Milestone
  - quote_date ← Entered Date
  - state ← State
  - product_family ← Part Description
- Extras kept for future use: Expected Close, Cloud Inside Rep, Territory, MonthlyRecurringCharge, Entered By, Last Changed Date, Expire Date, Summary, Note, Part Number, Quantity, UnitCost

## Features
- Regions + clickable drilldowns (week, account, vendor, state)
- KPIs, weekly trend, owner leaderboards
- CSV export of filtered/drilled tables
- SQLite persistence (swap to Turso/Supabase later if desired)
