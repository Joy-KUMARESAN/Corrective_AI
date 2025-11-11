# Corrective AI (Streamlit)

Streamlit app to **correct shipment addresses** (Google Geocoding) and **suggest HS codes** (SentenceTransformers).  
**Mistral via Ollama is required** for natural-language command parsing.

## Requirements
- **Python:** 3.10–3.12 (tested)
- **Ollama + Mistral:** install Ollama, then `ollama pull mistral`
- **Pip deps:** `pip install -r requirements.txt`

## Project Structure
corrective-ai/
├─ app.py
├─ .env
├─ requirements.txt
├─ data/
│ ├─ hs_codes_sample.csv
│ └─ original.csv
└─ src/corrective_ai/
├─ config.py
├─ logic/
│ ├─ address.py
│ ├─ hs_matching.py
│ └─ intent.py
├─ services/
│ ├─ geocode.py
│ ├─ embeddings.py
│ └─ llm.py
└─ ui/
├─ chat.py
└─ styles.py

shell
Copy code

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# LLM setup (required)
ollama pull mistral
Environment (.env)
env
Copy code
HS_CODE_CSV=data/hs_codes_sample.csv
GOOGLE_MAPS_API_KEY=YOUR_KEY
USE_LOCAL_LLM=true
LLM_MODEL_NAME=mistral
OLLAMA_BASE_URL=http://localhost:11434
Run
bash
Copy code
streamlit run app.py
Input CSV (minimum columns)
Item Name

Package Description

Destination Address 1, Destination Address 2, Destination Address 3

Destination City, Destination State, Destination ZIP

Destination Country, Destination Country Code

Error Description (and optional Error)

Commands (examples)
Batch address: “fix all addresses”

Single address (by item): “change the address for Blue Wallet”

Batch HS code: “fix all hs codes”

Single HS (by item): “update hs code for Blue Wallet”

Free-form address: “correct the address for 123 Main St, Springfield, IL”

Review & Apply: per-row OK/No + Accept All / Reject All.

Notes
Address issues are detected from Error Description (e.g., “Invalid Postal Code”, “Address1 is empty / has only special characters”, “OriginCountry is empty / has only special characters”). Extend patterns in logic/address.py.

HS suggestions use all-MiniLM-L6-v2 embeddings.

Mistral (via Ollama) is used for intent parsing and must be available.

Quick Troubleshooting
HS_CODE_CSV not found: check path in .env.

Model load error (Torch): use the pinned versions in requirements.txt.

Ollama errors: ensure ollama pull mistral completed and Ollama is running.

Copy code
