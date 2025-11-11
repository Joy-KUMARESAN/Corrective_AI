# Corrective AI — Address Correction (Streamlit)

A lightweight Streamlit app to **detect address issues** in your CSV (via `Error Description`), **propose corrected addresses** using Google Geocoding, and let users **approve updates** row-by-row or apply all at once.

---

## ✨ Features

- **Batch correction** – `fix all addresses` proposes corrections for every row with address-related errors.
- **Single-row correction**
  - by **Item Name**: `change the address for Blue Wallet`
  - by **Reference**: `correct the address for EPG011042500542210 - Ref. #`
  - by **Free-form address**: `correct the address for 123 Main St, Chicago IL` (returns a normalized address in chat)
- **Review & apply**
  - Proposals table with **Approve** checkbox per row
  - **Apply Selected**, **Accept All**, **Reject All**
  - Clears `Error` / `Error Description` once applied
- **Fast & safe**
  - Persistent **SQLite geocode cache**
  - Simple **QPS limit** for API calls

---

## 📄 Input CSV

Required (or auto-created) columns:

- `Item Name`
- `Destination Address 1`, `Destination Address 2`, `Destination Address 3`
- `Destination City`, `Destination State`, `Destination ZIP`
- `Destination Country`, `Destination Country Code`
- `Error Description` *(used to decide which rows need correction)*
- *(Optional)* `Error`

> To target single rows by a reference value, include a column like **`Ref. #`** (common variants such as `Reference Number`, `Tracking Number` are auto-detected).

**Include a tiny example file** (e.g., `sample_input.csv`) in the repo so others can try the UI quickly.

---

## 🚀 Quickstart

### 1) Requirements
- **Python 3.11+**
- **Google Maps Geocoding API key**

### 2) Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
3) Configure environment
Create a .env in the repo root:

env
Copy code
GOOGLE_MAPS_API_KEY=YOUR_KEY_HERE

# Optional (defaults shown)
GEOCODE_CACHE_DB=geocode_cache.sqlite
GEOCODE_CACHE_TTL_SECS=86400
GEOCODE_MAX_QPS=5
AUDIT_LOG=audit_log.csv
4) Run
bash
Copy code
streamlit run app.py
Open the URL Streamlit prints (e.g., http://localhost:8501).

🧭 Using the App
Upload your CSV (sidebar).

Type a command (lenient, conversational is fine). Examples:

fix all addresses

change the address for Blue Wallet

correct the address for EPG011042500542210 - Ref. #

correct the address for 123 Main St, Springfield, IL

Review the Proposals tab:

Tick Approve on rows you want.

Click Apply Selected or Accept All / Reject All.

The Results tab shows your live DataFrame; Download the updated CSV from the sidebar.

🔎 How errors are detected
Rows are considered address-related if Error Description matches patterns such as:

“Invalid Postal Code / ZIP”

“Address1 is empty / has only special characters”

“Origin/Destination Country/City/State is missing/invalid”

Non-address phrases (e.g., “has only special characters in item”) are ignored.

You can override patterns in .env (semicolon ; separated regular expressions):

env
Copy code
ADDR_PATTERNS=\binvalid\s*(postal\s*code|zip|zip\s*code|zipcode)\b;...
NEG_ADDR_PATTERNS=\bhas\s+only\s+special\s+characters\s+in\s+item\b
⚙️ Caching & Rate Limits
Geocode results cached in geocode_cache.sqlite (configurable).

Cache TTL default 24h (GEOCODE_CACHE_TTL_SECS).

API calls throttled by GEOCODE_MAX_QPS (default 5).

🧱 Folder Layout
arduino
Copy code
.
├─ app.py
├─ requirements.txt
├─ .env                  # not committed (contains API key)
├─ sample_input.csv      # small example file
└─ src/
   └─ utils/
      ├─ persistent_cache.py   # SQLite get/set by key
      └─ rate_limit.py         # simple QPS limiter
🛠️ Troubleshooting
No proposals appear → Check that Error Description contains address-related issues (see patterns above).

Single row not found → Use the hint suffix for references:

correct the address for EPG011042500542210 - Ref. #

Ensure your CSV has a matching reference column/value.

API errors → Verify GOOGLE_MAPS_API_KEY, network access, and Google Cloud quotas.

🌐 Deploying (brief)
Provision a small VM (Linux/Windows), install Python 3.11+, then:

bash
Copy code
pip install -r requirements.txt
Add .env with your API key, then:

bash
Copy code
streamlit run app.py
Keep it running behind a reverse proxy (nginx/IIS) or a process manager (screen, pm2, systemd).

Secure with firewall rules and HTTPS termination.

