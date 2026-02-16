import os
import sys
import re
import html
import hashlib
import asyncio
import csv
from datetime import datetime, timezone
import difflib
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.persistent_cache import cache_get, cache_set
from src.utils.rate_limit import respect_qps
from src.corrective_ai.logic.address import parse_address_components


# ---------- Boot / ENV ----------
print("[BOOT] Starting Corrective AI — Address Correction", flush=True)
load_dotenv()
print("[BOOT] .env loaded (if present)", flush=True)

st.set_page_config(page_title="Corrective AI — Address Correction", layout="wide")
print("[BOOT] Streamlit page configured", flush=True)

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    print("[BOOT] Windows event loop policy set", flush=True)

USE_LOCAL_LLM = (os.getenv("USE_LOCAL_LLM", "false").lower() in ["1", "true", "yes"])
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

GEOCODE_CACHE_DB = os.getenv("GEOCODE_CACHE_DB", "geocode_cache.sqlite")
GEOCODE_CACHE_TTL = int(os.getenv("GEOCODE_CACHE_TTL_SECS", str(24 * 3600)))
GEOCODE_MAX_QPS = int(os.getenv("GEOCODE_MAX_QPS", "5"))

AUDIT_LOG = os.getenv("AUDIT_LOG", "audit_log.csv")

print(f"[ENV] USE_LOCAL_LLM={USE_LOCAL_LLM}  LLM_MODEL_NAME={LLM_MODEL_NAME}", flush=True)
print(f"[ENV] OLLAMA_BASE_URL={OLLAMA_BASE_URL}", flush=True)
print(f"[ENV] GOOGLE_MAPS_API_KEY present? {'YES' if bool(GOOGLE_MAPS_API_KEY) else 'NO'}", flush=True)
print(f"[ENV] GEOCODE_CACHE_DB={GEOCODE_CACHE_DB}  TTL={GEOCODE_CACHE_TTL}s  QPS={GEOCODE_MAX_QPS}", flush=True)
print(f"[ENV] AUDIT_LOG={AUDIT_LOG}", flush=True)

SEARCH_COLS = [
    "Item Name",
    "Reference Number", "ReferenceNumber", "Reference", "Ref. #", "Ref.#", "Ref #",
    "EPG", "EPG Reference", "Tracking Number", "TrackingNumber",
    "SKU", "Item ID", "Item Code"
]

CANONICAL_FIELDS = [
    "Destination Address 1",
    "Destination Address 2",
    "Destination Address 3",
    "Destination City",
    "Destination State",
    "Destination ZIP",
    "Destination Country",
    "Destination Country Code",
]

HEADER_VARIANTS = {
    "Destination Address 1": [
        "destination address 1", "address 1", "addr1", "address line 1",
        "destination addr 1", "ship to address 1", "ship address 1", "address1"
    ],
    "Destination Address 2": [
        "destination address 2", "address 2", "addr2", "address line 2",
        "destination addr 2", "ship to address 2", "ship address 2", "address2"
    ],
    "Destination Address 3": [
        "destination address 3", "address 3", "addr3", "address line 3",
        "destination addr 3", "ship to address 3", "ship address 3", "address3"
    ],
    "Destination City": [
        "destination city", "city", "ship to city", "ship city"
    ],
    "Destination State": [
        "destination state", "state", "province", "region", "state/province", "state code",
        "ship to state", "ship state"
    ],
    "Destination ZIP": [
        "destination zip", "zip", "zip code", "postal code", "postcode",
        "zip/postal code", "zip or postal code", "postal", "zip/postal",
        "ship to zip", "ship zip", "ship postal code", "zip code/postal code", "zip/postcode"
    ],
    "Destination Country": [
        "destination country", "country", "ship to country", "ship country"
    ],
    "Destination Country Code": [
        "destination country code", "country code", "iso country", "iso2", "country iso", "countrycode"
    ],
}

ERROR_COL_VARIANTS = [
    "Error Description", "Error/Remarks", "Error Remarks", "Remarks", "Error", "Error Message"
]

# ---------- Header resolution helpers ----------
def normalize_header(h: str) -> str:
    if h is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(h).strip().lower())

def resolve_header_map(df: pd.DataFrame) -> dict:
    existing = {normalize_header(c): c for c in df.columns}
    colmap = {}
    for canonical in CANONICAL_FIELDS:
        variants = [canonical] + HEADER_VARIANTS.get(canonical, [])
        found = None
        for v in variants:
            key = normalize_header(v)
            if key in existing:
                found = existing[key]
                break
        colmap[canonical] = found if found else canonical
    print(f"[HEADERS] Resolved column map: {colmap}", flush=True)
    return colmap

def resolve_error_col(df: pd.DataFrame) -> str:
    existing = {normalize_header(c): c for c in df.columns}
    for v in ERROR_COL_VARIANTS:
        key = normalize_header(v)
        if key in existing:
            print(f"[HEADERS] Found error column: {existing[key]}", flush=True)
            return existing[key]
    if "Error Description" not in df.columns:
        df["Error Description"] = ""
        print("[HEADERS] Created default 'Error Description' column", flush=True)
    return "Error Description"

def get_val(row: pd.Series, colmap: dict, canonical: str) -> str:
    actual = colmap.get(canonical, canonical)
    return (row.get(actual, "") or "").strip()

def set_val(df: pd.DataFrame, idx, colmap: dict, canonical: str, value: str):
    actual = colmap.get(canonical, canonical)
    if actual not in df.columns:
        df[actual] = ""
        print(f"[WRITE] Created missing column '{actual}'", flush=True)
    df.at[idx, actual] = value
    print(f"[WRITE] Row {idx}: {actual} <- {value}", flush=True)

# ---------- Merge helper ----------
def apply_components_merge(df: pd.DataFrame, idx: int, colmap: dict, comps: dict) -> dict:
    changed = {}
    address_fields  = ["Destination Address 1", "Destination Address 2", "Destination Address 3"]
    location_fields = ["Destination City", "Destination State", "Destination ZIP",
                       "Destination Country", "Destination Country Code"]

    # Address lines: only if API value provided
    for canonical in address_fields:
        api_val = (comps.get(canonical, "") or "").strip()
        if api_val:
            prev = get_val(df.loc[idx], colmap, canonical)
            if str(prev) != api_val:
                set_val(df, idx, colmap, canonical, api_val)
                changed[canonical] = api_val
                print(f"[MERGE] Row {idx}: updated address field '{canonical}'", flush=True)
        else:
            print(f"[MERGE] Row {idx}: skipped blank API value for '{canonical}'", flush=True)

    for canonical in location_fields:
        api_val = (comps.get(canonical, "") or "").strip()
        if api_val:
            prev = get_val(df.loc[idx], colmap, canonical)
            if str(prev) != api_val:
                set_val(df, idx, colmap, canonical, api_val)
                changed[canonical] = api_val
                print(f"[MERGE] Row {idx}: updated location field '{canonical}'", flush=True)
        else:
            print(f"[MERGE] Row {idx}: kept original for '{canonical}' (API blank)", flush=True)

    return changed

# ---------- Item resolver ----------
def find_item_row(df: pd.DataFrame, query: str, preferred_col: str | None = None):
    q = (query or "").strip().lower()
    if not q or df.empty:
        print("[FIND] Empty query or DF; skipping", flush=True)
        return None, None

    if preferred_col and preferred_col in df.columns:
        print(f"[FIND] Trying preferred column: {preferred_col}", flush=True)
        series = df[preferred_col].astype(str).fillna("").str.strip()
        exact_hits = df.index[series.str.lower() == q].tolist()
        if exact_hits:
            idx = int(exact_hits[0])
            name = df.at[idx, "Item Name"] if "Item Name" in df.columns else series.loc[idx]
            print(f"[FIND] Exact hit in preferred col. idx={idx} name={name}", flush=True)
            return idx, name
        contains_hits = df.index[series.str.lower().str.contains(re.escape(q))].tolist()
        if contains_hits:
            idx = int(contains_hits[0])
            name = df.at[idx, "Item Name"] if "Item Name" in df.columns else series.loc[idx]
            print(f"[FIND] Contains hit in preferred col. idx={idx} name={name}", flush=True)
            return idx, name

    for col in SEARCH_COLS:
        if col in df.columns:
            series = df[col].astype(str).fillna("").str.strip()
            exact_hits = df.index[series.str.lower() == q].tolist()
            if exact_hits:
                idx = int(exact_hits[0])
                name = df.at[idx, "Item Name"] if "Item Name" in df.columns else series.loc[idx]
                print(f"[FIND] Exact hit in {col}. idx={idx} name={name}", flush=True)
                return idx, name
            contains_hits = df.index[series.str.lower().str.contains(re.escape(q))].tolist()
            if contains_hits:
                idx = int(contains_hits[0])
                name = df.at[idx, "Item Name"] if "Item Name" in df.columns else series.loc[idx]
                print(f"[FIND] Contains hit in {col}. idx={idx} name={name}", flush=True)
                return idx, name

    candidates = []
    for col in SEARCH_COLS:
        if col in df.columns:
            candidates.extend(df[col].astype(str).fillna("").str.strip().tolist())
    candidates = [c for c in set(candidates) if c]

    best = difflib.get_close_matches(q, [c.lower() for c in candidates], n=1, cutoff=0.6)
    if best:
        target = best[0]
        for col in SEARCH_COLS:
            if col in df.columns:
                hits = df.index[df[col].astype(str).str.strip().str.lower() == target].tolist()
                if hits:
                    idx = int(hits[0])
                    name = df.at[idx, "Item Name"] if "Item Name" in df.columns else df.at[idx, col]
                    print(f"[FIND] Fuzzy hit in {col}. idx={idx} name={name}", flush=True)
                    return idx, name
    print("[FIND] No match found", flush=True)
    return None, None

# ---------- HTTP SESSION ----------
def _build_session():
    print("[HTTP] Building session with retries", flush=True)
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
    s.mount("http://",  HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
    return s

HTTP = _build_session()

# ---------- CSS ----------
st.markdown("""
<style>
body, .stApp { background: #f7f8fb; color: #1f2937; }
[data-testid="stSidebar"] { background: #f2f5fb; }
.chat-bubble-bot { background: #e9eef9; color: #111827; border-left: 6px solid #3b82f6;
  border-radius: 8px; padding: 10px; margin-bottom: 8px; margin-right: 30%; }
.chat-bubble-user { background: #3b82f6; color: #fff; border-radius: 8px; padding: 10px;
  margin-bottom: 8px; margin-left: 30%; text-align: right; }
[data-testid="stDataEditor"] table { font-size: 0.94rem; }
</style>
""", unsafe_allow_html=True)
print("[UI] CSS injected", flush=True)

# ---------- Address error detection ----------
def _env_patterns(key: str, defaults: list[str]) -> list[str]:
    raw = (os.getenv(key, "") or "").strip()
    return [p.strip() for p in raw.split(";") if p.strip()] if raw else defaults

_DEFAULT_ADDRESS_ERROR_PATTERNS = [
    r"\binvalid\s*(postal\s*code|zip|zip\s*code|zipcode)\b",
    r"\b(postal\s*code|zip\s*code)\s*invalid\b",
    r"\baddress\s*1\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\baddress1\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\baddress(?:\s*line)?\s*1\s*(?:is\s*)?(?:empty|missing)\b",
    r"\binvalid\s*country\s*code\b",
    r"\b(country\s*code)\s*(?:is\s*)?(invalid|empty|missing)\b",
    r"\b(city|state|province|country)\s*(?:is\s*)?(invalid|empty|missing)\b",
]
_DEFAULT_NEGATIVE_PATTERNS = [
    r"\bhas\s+only\s+special\s+characters\s+in\s+item\b",
]

ADDRESS_ERROR_PATTERNS = _env_patterns("ADDR_PATTERNS", _DEFAULT_ADDRESS_ERROR_PATTERNS)
NEGATIVE_PATTERNS      = _env_patterns("NEG_ADDR_PATTERNS", _DEFAULT_NEGATIVE_PATTERNS)
print(f"[PATTERNS] ADDRESS={ADDRESS_ERROR_PATTERNS}", flush=True)
print(f"[PATTERNS] NEGATIVE={NEGATIVE_PATTERNS}", flush=True)

def is_address_error(text: str) -> bool:
    if not text:
        return False
    for pat in NEGATIVE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            print(f"[ERRCHK] Negative pattern matched: {pat}", flush=True)
            return False
    for pat in ADDRESS_ERROR_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            print(f"[ERRCHK] Address-error pattern matched: {pat}", flush=True)
            return True
    return False
def normalize_address_for_google(
    addr1: str,
    addr2: str,
    addr3: str,
    city: str,
    state: str,
    zip_code: str,
    country: str,
) -> str:
    """
    Build a clean, Google-friendly address string.
    - Deduplicates address lines
    - Drops ZIP (Google should infer it)
    - Preserves house number + route
    """

    parts = []
    seen = set()

    for p in [addr1, addr2, addr3]:
        p = (p or "").strip()
        key = p.lower()
        if p and key not in seen:
            parts.append(p)
            seen.add(key)

    if city:
        parts.append(city.strip())

    if state:
        parts.append(state.strip())

    if country:
        parts.append(country.strip())

    normalized = ", ".join(parts)

    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r",\s*,+", ", ", normalized)

    print(f"[NORMALIZE] Google input => {normalized}", flush=True)
    return normalized

# ---------- Cache helpers ----------
def normalize_address_for_cache(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------- Geocoding ----------
def correct_address(address: str):
    print(f"[GEO] correct_address called for: {address}", flush=True)
    if not GOOGLE_MAPS_API_KEY:
        print("[GEO][ERROR] Missing GOOGLE_MAPS_API_KEY", flush=True)
        return {"api_error": "Google Maps API key is missing. Set GOOGLE_MAPS_API_KEY in .env"}

    addr = (address or "").strip()
    if not addr or re.fullmatch(r"[\W_]+", addr):
        print("[GEO][ERROR] Address empty or malformed", flush=True)
        return {"error_description_specific": "Address input is empty or malformed."}

    hit = cache_get(GEOCODE_CACHE_DB, addr, ttl_secs=GEOCODE_CACHE_TTL)
    if hit is not None:
        print("[GEO] Cache HIT", flush=True)
        return hit
    print("[GEO] Cache MISS", flush=True)

    url = (
    f"https://maps.googleapis.com/maps/api/geocode/json"
    f"?address={addr}"
    f"&language=en"
    f"&key={GOOGLE_MAPS_API_KEY}"
)
    try:
        respect_qps(GEOCODE_MAX_QPS)
        resp = HTTP.get(url, timeout=12)
        print(f"[GEO] HTTP status={resp.status_code}", flush=True)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        results = data.get("results", [])
        print(f"[GEO] API status={status} results={len(results)}", flush=True)
        if status == "OK" and results:
            top = results[0]
            result = {
                "formatted_address": top.get("formatted_address", ""),
                "components": top.get("address_components", [])
            }
            cache_set(GEOCODE_CACHE_DB, addr, result)
            print("[GEO] Parsed OK, cached result", flush=True)
            return result
        if status == "ZERO_RESULTS":
            result = {"error_description_specific": "No valid address found for the given input."}
            cache_set(GEOCODE_CACHE_DB, addr, result)
            print("[GEO] ZERO_RESULTS cached", flush=True)
            return result
        result = {"error_description_specific":
                  f"Google Maps API error: {status or 'UNKNOWN_STATUS'} - {data.get('error_message', 'No message')}"}
        cache_set(GEOCODE_CACHE_DB, addr, result)
        print("[GEO][ERROR] Non-OK status cached", flush=True)
        return result
    except requests.exceptions.Timeout:
        print("[GEO][ERROR] Timeout", flush=True)
        return {"error_description_specific": "Address correction request timed out."}
    except requests.exceptions.RequestException as e:
        print(f"[GEO][ERROR] Request exception: {e}", flush=True)
        return {"error_description_specific": f"Network or API request error: {e}"}
    except Exception as e:
        print(f"[GEO][ERROR] Unexpected: {e}", flush=True)
        return {"error_description_specific": f"Address correction processing error: {e}"}

def ollama_chat(prompt: str) -> str:
    try:
        print("[LLM] Calling Ollama", flush=True)
        r = HTTP.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=30
        )
        r.raise_for_status()
        resp = (r.json().get("response") or "").strip()
        print("[LLM] Response received", flush=True)
        return resp
    except Exception as e:
        print(f"[LLM][ERROR] {e}", flush=True)
        return f"[LLM error: {e}]"

ADDR_WORDS = [
    "address", "addresses",
    "destination address", "destination addresses",
    "normalize address", "normalize addresses",
    "fix address", "fix addresses",
    "clean address", "clean addresses",
    "correct address", "correct addresses",
    "change address", "change addresses",
]
BATCH_WORDS = ["all", "every", "entire file", "bulk", "batch", "everything", "all rows"]
SINGLE_VERBS = ["fix", "correct", "change", "update", "modify", "edit", "set"]

def _contains_any(text: str, words) -> bool:
    t = text.lower()
    return any(w in t for w in words)

def _extract_item_name(text: str, item_names):
    t = text.lower()
    for name in sorted(item_names, key=len, reverse=True):
        if name and name.lower() in t:
            return name
    return None

def _extract_address_string(text: str):
    m = re.search(r"(?:correct|fix|update|change)\s+the\s+address\s+for\s+(?:this\s+)?(.+)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def _extract_hint_column(text: str) -> str | None:
    t = text.lower()
    if ("ref" in t) or ("ref. #" in t) or ("reference number" in t) or ("reference" in t):
        return "Ref. #"
    if "epg" in t or "tracking" in t:
        return "EPG" if "EPG" in SEARCH_COLS else "Tracking Number"
    if "sku" in t:
        return "SKU"
    return None

def parse_user_intent(user_input: str, item_names):
    text = (user_input or "").strip()
    print(f"[INTENT] Raw input='{text}'", flush=True)
    if not text:
        return None, None, None

    matched_item = _extract_item_name(text, item_names)
    address_string = _extract_address_string(text)

    if _contains_any(text, ADDR_WORDS) and _contains_any(text, BATCH_WORDS):
        print("[INTENT] batch_correct_addresses", flush=True)
        return "batch_correct_addresses", None, None
    if _contains_any(text, ADDR_WORDS) and matched_item and any(v in text.lower() for v in SINGLE_VERBS):
        print(f"[INTENT] correct_address_for_item -> {matched_item}", flush=True)
        return "correct_address_for_item", matched_item, None
    if address_string:
        print(f"[INTENT] correct_single_address -> '{address_string}'", flush=True)
        return "correct_single_address", None, address_string

    if not USE_LOCAL_LLM:
        print("[INTENT] No match and LLM disabled", flush=True)
        return None, None, None

    prompt = f"""
Classify into one of: batch_correct_addresses | correct_address_for_item | correct_single_address.
Return JSON with keys: intent, item_name, address_string.
Items: {list(item_names)}
User: \"\"\"{text}\"\"\" 
JSON:
"""
    raw = ollama_chat(prompt)
    if raw.startswith("[LLM error:"):
        print("[INTENT] LLM error path", flush=True)
        return None, None, None
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    blob = m.group(0) if m else raw
    try:
        import json
        data = json.loads(blob)
        intent = data.get("intent")
        item = data.get("item_name") or matched_item
        addr = data.get("address_string")
        print(f"[INTENT] LLM intent={intent} item={item} addr={addr}", flush=True)
        if intent not in {"batch_correct_addresses", "correct_address_for_item", "correct_single_address"}:
            return None, None, None
        if intent == "correct_address_for_item" and not item:
            item = _extract_item_name(text, item_names)
        return intent, item, addr
    except Exception as e:
        print(f"[INTENT][ERROR] {e}", flush=True)
        return None, None, None

def chat_bubble(sender, message):
    if sender == "bot":
        st.markdown(f'<div class="chat-bubble-bot">{html.escape(message)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-user">{html.escape(message)}</div>', unsafe_allow_html=True)

def audit_rows(applied_props: list[dict], df: pd.DataFrame):
    if not applied_props:
        print("[AUDIT] No rows to audit", flush=True)
        return
    path = AUDIT_LOG
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["ts","user","row_idx","item","field","old","new"])
            print(f"[AUDIT] Created audit log at {path}", flush=True)
        ts = datetime.now(timezone.utc).isoformat()
        user = st.session_state.get("username", "unknown")
        for p in applied_props:
            idx = p["idx"]; item = p.get("item_name","")
            comps = p.get("new_components", {})
            for canonical in CANONICAL_FIELDS:
                if canonical in comps:
                    actual = st.session_state.colmap.get(canonical, canonical)
                    old = df.at[idx, actual] if actual in df.columns else ""
                    new = comps.get(canonical, "")
                    if str(old) != str(new):
                        w.writerow([ts, user, idx, item, actual, old, new])
                        print(f"[AUDIT] Row {idx} {actual}: '{old}' -> '{new}'", flush=True)

async def async_main():
    print("[APP] async_main start", flush=True)
    st.title("Corrective AI — Address Correction")

    ss = st.session_state
    if "results_df" not in ss:
        ss.results_df = pd.DataFrame()
    if "results_version" not in ss:
        ss.results_version = 0
    if "chat_history" not in ss:
        ss.chat_history = []
    if "proposals" not in ss:
        ss.proposals = []
    if "proposals_df" not in ss:
        ss.proposals_df = pd.DataFrame()
    if "csv_loaded" not in ss:
        ss.csv_loaded = False
    if "review_msg" not in ss:
        ss.review_msg = ""
    if "colmap" not in ss:
        ss.colmap = {c: c for c in CANONICAL_FIELDS}
    if "errcol" not in ss:
        ss.errcol = "Error Description"

    # Sidebar
    with st.sidebar:
        st.header("Actions")
        uploaded = st.file_uploader(
            "Upload CSV (address columns + error column)",
            type=["csv"],
            help="We detect headers (e.g., Address1/City/Zip/Country) and update in place."
        )

        with st.form("cmd_form", clear_on_submit=True):
            user_input = st.text_input(
                "Command (lenient)",
                placeholder="e.g., fix all addresses / correct the address for EPG0110… (Ref. #) / correct the address for 123 Main St..."
            )
            submitted = st.form_submit_button("Send")
            if submitted:
                print(f"[CMD] Submitted command='{user_input}'", flush=True)

        if not ss.results_df.empty:
            st.download_button(
                "💾 Download Current CSV",
                data=ss.results_df.to_csv(index=False).encode("utf-8"),
                file_name="corrected_results.csv",
                mime="text/csv",
            )

        if st.button("Clear cache"):
            st.cache_data.clear()
            print("[CACHE] Streamlit cache cleared", flush=True)
            st.success("Streamlit cache cleared (persistent cache is separate).")

        if st.button("Reset"):
            print("[UI] Reset clicked", flush=True)
            ss.results_df = pd.DataFrame()
            ss.results_version = 0
            ss.chat_history = []
            ss.proposals = []
            ss.proposals_df = pd.DataFrame()
            ss.csv_loaded = False
            ss.review_msg = ""
            ss.colmap = {c: c for c in CANONICAL_FIELDS}
            ss.errcol = "Error Description"
            st.rerun()

        st.markdown("**Examples:**\n- fix all addresses\n- change the address for **EPG011042500542210** — Ref. #\n- change the address for **Blue Wallet** — Item Name\n- correct the address for **123 Main St, Springfield, IL**")

    # Tabs
    tab_proposals, tab_results, tab_chat = st.tabs(["Proposals", "Results", "Chat"])

    with tab_results:
        if not ss.results_df.empty:
            st.caption(f"Results (v{ss.results_version})")
            st.dataframe(ss.results_df)
            print(f"[UI] Showing results v{ss.results_version}, rows={len(ss.results_df)}", flush=True)

    with tab_chat:
        for sender, msg in ss.chat_history:
            chat_bubble(sender, msg)

    with tab_proposals:
        if not ss.proposals_df.empty:
            st.markdown("### Proposed Address Corrections")
            if ss.review_msg:
                st.info(ss.review_msg)

            # Ensure int Row Index for editor
            if "Row Index" in ss.proposals_df.columns:
                try:
                    ss.proposals_df["Row Index"] = ss.proposals_df["Row Index"].astype(int)
                except Exception:
                    ss.proposals_df["Row Index"] = ss.proposals_df["Row Index"].apply(
                        lambda x: int(float(x)) if pd.notna(x) else x
                    )

            column_config = {
                "Approve": st.column_config.CheckboxColumn("Approve", default=False, width="small"),
                "Row Index": st.column_config.NumberColumn("Row", step=1, format="%d", width="small"),
            }

            approve_first_order = ["Approve"] + [c for c in ss.proposals_df.columns if c != "Approve"]

            table_height = min(600, 140 + 32 * len(ss.proposals_df))
            edited_df = st.data_editor(
                ss.proposals_df,
                key="proposals_editor",
                num_rows="fixed",
                height=table_height,
                hide_index=True,
                column_config=column_config,
                column_order=approve_first_order,
                disabled=[c for c in ss.proposals_df.columns if c not in ("Approve",)],
            )
            st.session_state.proposals_df = edited_df
            print(f"[UI] Proposals table displayed, rows={len(ss.proposals_df)}", flush=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✅ Apply Selected", key="apply_selected", type="primary"):
                    print("[APPLY] Apply Selected clicked", flush=True)
                    df = ss.results_df
                    if "Approve" in edited_df.columns:
                        sel = edited_df[edited_df["Approve"] == True].copy()
                    else:
                        sel = edited_df.iloc[0:0].copy()
                    print(f"[APPLY] Selected rows={len(sel)}", flush=True)

                    if not sel.empty and "Row Index" in sel.columns:
                        try:
                            sel["Row Index"] = sel["Row Index"].astype(int)
                        except Exception:
                            sel["Row Index"] = sel["Row Index"].apply(lambda x: int(float(x)) if pd.notna(x) else x)
                        selected = set(sel["Row Index"].tolist())
                    else:
                        selected = set()
                    print(f"[APPLY] Selected indices={selected}", flush=True)

                    pmap = {int(p["idx"]): p for p in ss.proposals}
                    applied_props = []
                    for idx in selected:
                        p = pmap.get(int(idx))
                        if not p:
                            print(f"[APPLY] Missing proposal for idx={idx}", flush=True)
                            continue
                        comps = p.get("new_components", {})
                        changed = apply_components_merge(df, int(idx), ss.colmap, comps)
                        zip_col = ss.colmap.get("Destination ZIP")
                        if zip_col in df.columns:
                            df[zip_col] = df[zip_col].astype(str)

                        if changed:
                            if ss.errcol in df.columns:
                                df.at[int(idx), ss.errcol] = ""
                            if "Error" in df.columns and ss.errcol != "Error":
                                df.at[int(idx), "Error"] = ""
                            applied_props.append(p)
                            print(f"[APPLY] Applied to row {idx}", flush=True)
                        else:
                            print(f"[APPLY] Nothing changed for row {idx}", flush=True)

                    audit_rows(applied_props, df)

                    ss.proposals = [p for p in ss.proposals if int(p["idx"]) not in selected]
                    if ss.proposals:
                        rem_df = edited_df[~edited_df["Row Index"].isin(list(selected))].copy()
                        rem_df["Approve"] = False
                        ss.proposals_df = rem_df
                    else:
                        ss.proposals_df = pd.DataFrame()

                    ss.results_df = df.copy(deep=True)
                    ss.results_version += 1
                    ss.review_msg = f"Applied {len(applied_props)} selected row(s)."
                    print(f"[APPLY] Done. v{ss.results_version}", flush=True)
                    st.rerun()

            with c2:
                if st.button("✅ Accept All Changes", key="accept_all"):
                    print("[APPLY] Accept All clicked", flush=True)
                    df = ss.results_df
                    applied_props = []
                    for p in ss.proposals:
                        idx = int(p["idx"])
                        comps = p.get("new_components", {})
                        changed = apply_components_merge(df, idx, ss.colmap, comps)
                        if changed:
                            if ss.errcol in df.columns:
                                df.at[idx, ss.errcol] = ""
                            if "Error" in df.columns and ss.errcol != "Error":
                                df.at[idx, "Error"] = ""
                            applied_props.append(p)
                            print(f"[APPLY] Accepted for row {idx}", flush=True)
                        else:
                            print(f"[APPLY] Skipped (no change) row {idx}", flush=True)

                    audit_rows(applied_props, df)

                    ss.results_df = df.copy(deep=True)
                    ss.results_version += 1
                    ss.proposals = []
                    ss.proposals_df = pd.DataFrame()
                    ss.review_msg = "Applied all proposed address corrections."
                    print(f"[APPLY] Accept All complete. v{ss.results_version}", flush=True)
                    st.rerun()

            with c3:
                if st.button("❌ Reject All Changes", key="reject_all"):
                    print("[APPLY] Reject All clicked", flush=True)
                    ss.proposals = []
                    ss.proposals_df = pd.DataFrame()
                    ss.review_msg = "Discarded all proposed corrections."
                    st.rerun()
        else:
            st.info("No proposed corrections yet. Run a command like “fix all addresses”.")
            if len(ss.proposals_df) == 0:
                print("[UI] Proposals empty", flush=True)

    # ---------- Handle CSV upload ----------
    if 'uploaded' in locals() and uploaded and not ss.csv_loaded:
        print("[UPLOAD] File uploaded; reading...", flush=True)
        try:
            df = pd.read_csv(uploaded, keep_default_na=False).fillna('')
            print(f"[UPLOAD] CSV read OK, rows={len(df)} cols={len(df.columns)}", flush=True)

            # Resolve headers
            ss.colmap = resolve_header_map(df)
            ss.errcol = resolve_error_col(df)
            print(f"[UPLOAD] errcol='{ss.errcol}'", flush=True)

            # ✅ FORCE Destination ZIP column to STRING (CRITICAL FIX)
            zip_col = ss.colmap.get("Destination ZIP")
            if zip_col in df.columns:
                df[zip_col] = df[zip_col].astype(str)
                print(f"[UPLOAD] Forced ZIP column '{zip_col}' to string", flush=True)

            # Ensure Item Name exists
            if "Item Name" not in df.columns:
                df["Item Name"] = ""
                print("[UPLOAD] Created missing 'Item Name' column", flush=True)

            # Store results
            ss.results_df = df.copy(deep=True)
            ss.results_version = 0
            ss.csv_loaded = True
            print("[UPLOAD] Stored DF in session; rerun", flush=True)

            ss.chat_history.append((
                "bot",
                f"CSV uploaded. Detected error column: **{ss.errcol}**. "
                "You can now say 'fix all addresses' or "
                "'correct the address for EPG… (Ref. #)'."
            ))

            st.rerun()

        except Exception as e:
            print(f"[UPLOAD][ERROR] {e}", flush=True)
            st.error(f"Failed to read uploaded CSV: {e}")

    # ---------- Handle commands ----------
    if 'submitted' in locals() and submitted and (user_input or "").strip():
        print("[CMD] Handling command", flush=True)
        ss.chat_history.append(("user", user_input))
        df = ss.results_df

        if df.empty:
            print("[CMD] No DF; ask for CSV", flush=True)
            ss.chat_history.append(("bot", "Please upload a CSV first."))
            st.rerun()

        item_names = list(df["Item Name"].dropna().unique()) if "Item Name" in df.columns else []
        intent, matched_item, address_string_or_token = parse_user_intent(user_input, item_names)
        print(f"[CMD] intent={intent} matched_item={matched_item} address_token={address_string_or_token}", flush=True)

        # Batch correction — ONLY when error text matches address-error patterns
        if intent == "batch_correct_addresses":
            print("[BATCH] Enter", flush=True)
            ss.proposals = []
            ss.proposals_df = pd.DataFrame()
            with st.spinner("Correcting addresses in batch..."):
                addr_map = {}
                total_scanned = 0
                for idx, row in df.iterrows():
                    total_scanned += 1
                    err_text = (row.get(ss.errcol, "") or "").strip()
                    if not is_address_error(err_text):
                        continue
                    original_full = normalize_address_for_google(
                        addr1=get_val(row, ss.colmap, "Destination Address 1"),
                        addr2=get_val(row, ss.colmap, "Destination Address 2"),
                        addr3=get_val(row, ss.colmap, "Destination Address 3"),
                        city=get_val(row, ss.colmap, "Destination City"),
                        state=get_val(row, ss.colmap, "Destination State"),
                        zip_code=get_val(row, ss.colmap, "Destination ZIP"),  # intentionally ignored
                        country=get_val(row, ss.colmap, "Destination Country"),
                    )
                    if not original_full or re.fullmatch(r"[\W_]+", original_full):
                        continue

                    key = normalize_address_for_cache(original_full)
                    addr_map.setdefault(key, {"full": original_full, "rows": []})["rows"].append(int(idx))

                print(f"[BATCH] Scanned={total_scanned} unique_addr_groups={len(addr_map)}", flush=True)
                display_rows = []
                for key, bundle in addr_map.items():
                    corr = correct_address(bundle["full"])
                    if "formatted_address" not in corr:
                        print("[BATCH] Skip group (no formatted_address)", flush=True)
                        continue
                    comps = parse_address_components(corr["components"])
                    zip_val = comps.get("Destination ZIP", "").strip()
                    if not zip_val:
                        print("[BATCH] Warning: no ZIP from Google, keeping original", flush=True)
                    for idx in bundle["rows"]:
                        row = df.loc[int(idx)]
                        ss.proposals.append({
                            "idx": int(idx),
                            "item_name": row.get("Item Name", f"Row {int(idx)+1}"),
                            "original_full": bundle["full"],
                            "formatted_new_address": corr["formatted_address"],
                            "new_components": comps
                        })
                        display_rows.append({
                            "Row Index": int(idx),
                            "Item Name": row.get("Item Name", f"Row {int(idx)+1}"),
                            "Original Full Address": bundle["full"],
                            "Proposed Full Address": corr["formatted_address"],
                            "Proposed Address 1": comps.get("Destination Address 1", ""),
                            "Proposed Address 2": comps.get("Destination Address 2", ""),
                            "Proposed Address 3": comps.get("Destination Address 3", ""),
                            "Proposed City": comps.get("Destination City", ""),
                            "Proposed State": comps.get("Destination State", ""),
                            "Proposed ZIP": comps.get("Destination ZIP", ""),
                            "Proposed Country": comps.get("Destination Country", ""),
                            "Proposed Country Code": comps.get("Destination Country Code", ""),
                            "Approve": False,
                        })
                ss.proposals_df = pd.DataFrame(display_rows)

            ss.review_msg = f"Batch address correction completed. {len(ss.proposals)} row(s) proposed."
            print(f"[BATCH] Complete. proposals={len(ss.proposals)}", flush=True)
            ss.chat_history.append(("bot", ss.review_msg))
            st.rerun()

        # Single (identifier or literal) — ONLY when that row's error text matches patterns
        elif intent in {"correct_address_for_item", "correct_single_address"} and address_string_or_token:
            print("[SINGLE] Enter", flush=True)
            hint_col = _extract_hint_column(user_input)
            idx, resolved_name = find_item_row(df, address_string_or_token, preferred_col=hint_col)
            if idx is not None:
                row = df.loc[int(idx)]
                err_text = (row.get(ss.errcol, "") or "").strip()
                if not is_address_error(err_text):
                    print(f"[SINGLE] No address-error for '{resolved_name}'", flush=True)
                    ss.chat_history.append(("bot", f"No address-related issue detected for '{resolved_name}'."))
                    st.rerun()

                original_full = normalize_address_for_google(
                    addr1=get_val(row, ss.colmap, "Destination Address 1"),
                    addr2=get_val(row, ss.colmap, "Destination Address 2"),
                    addr3=get_val(row, ss.colmap, "Destination Address 3"),
                    city=get_val(row, ss.colmap, "Destination City"),
                    state=get_val(row, ss.colmap, "Destination State"),
                    zip_code=get_val(row, ss.colmap, "Destination ZIP"),  # intentionally ignored
                    country=get_val(row, ss.colmap, "Destination Country"),
                )


                with st.spinner(f"Correcting address for '{resolved_name}'..."):
                    corr = correct_address(original_full)
                    if "formatted_address" in corr:
                        comps = parse_address_components(corr["components"])
                        if comps.get("Destination ZIP", "").strip():
                            ss.proposals = [{
                                "idx": int(idx),
                                "item_name": resolved_name,
                                "original_full": original_full,
                                "formatted_new_address": corr["formatted_address"],
                                "new_components": comps
                            }]
                            ss.proposals_df = pd.DataFrame([{
                                "Row Index": int(idx),
                                "Item Name": resolved_name,
                                "Original Full Address": original_full,
                                "Proposed Full Address": corr["formatted_address"],
                                "Proposed Address 1": comps.get("Destination Address 1", ""),
                                "Proposed Address 2": comps.get("Destination Address 2", ""),
                                "Proposed Address 3": comps.get("Destination Address 3", ""),
                                "Proposed City": comps.get("Destination City", ""),
                                "Proposed State": comps.get("Destination State", ""),
                                "Proposed ZIP": comps.get("Destination ZIP", ""),
                                "Proposed Country": comps.get("Destination Country", ""),
                                "Proposed Country Code": comps.get("Destination Country Code", ""),
                                "Approve": False,
                            }])
                            ss.review_msg = f"Proposed correction for '{resolved_name}'."
                            print("[SINGLE] Proposal created for selected row", flush=True)
                        else:
                            print("[SINGLE] No ZIP in corrected address", flush=True)
                            ss.chat_history.append(("bot", "Corrected address has no ZIP; not proposed."))
                    else:
                        err = corr.get("api_error") or corr.get("error_description_specific")
                        print(f"[SINGLE][ERROR] {err}", flush=True)
                        ss.chat_history.append(("bot", f"Address correction failed: {err}"))
                st.rerun()
            else:
                print("[SINGLE] No row matched; treating as literal normalize", flush=True)
                with st.spinner(f"Correcting: {address_string_or_token}"):
                    corr = correct_address(address_string_or_token)
                if "formatted_address" in corr:
                    comps = parse_address_components(corr["components"])
                    pretty = [comps.get("Destination Address 1", ""),
                              comps.get("Destination Address 2", ""),
                              comps.get("Destination Address 3", ""),
                              comps.get("Destination City", ""),
                              comps.get("Destination State", ""),
                              comps.get("Destination Country", "")]
                    ss.chat_history.append(("bot",
                        f"Formatted: {corr['formatted_address']} \nWithout ZIP: {', '.join([p for p in pretty if p])}"))
                    print("[SINGLE] Literal normalize result shown in chat", flush=True)
                else:
                    err = corr.get("api_error") or corr.get("error_description_specific")
                    ss.chat_history.append(("bot", f"Could not correct: {err}"))
                    print(f"[SINGLE][ERROR] {err}", flush=True)
                st.rerun()
        else:
            print("[CMD] Unrecognized command path", flush=True)
            ss.chat_history.append(("bot",
                "I couldn't understand that command. Try:\n"
                "- **fix all addresses**\n"
                "- **change the address for EPG0110… (Ref. #)**\n"
                "- **correct the address for 123 Main St, Springfield, IL**"
            ))
            st.rerun()

    print("[APP] async_main end", flush=True)

if __name__ == "__main__":
    print("[MAIN] Running app", flush=True)
    asyncio.run(async_main())
