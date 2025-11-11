import os
import sys
import re
import html
import hashlib
import asyncio
import csv
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.persistent_cache import cache_get, cache_set
from src.utils.rate_limit import respect_qps

# ---------- ENV & PAGE ----------
load_dotenv()
st.set_page_config(page_title="Corrective AI — Address Correction", layout="wide")

# Windows event loop policy
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ---------- CONFIG ----------
USE_LOCAL_LLM = (os.getenv("USE_LOCAL_LLM", "false").lower() in ["1", "true", "yes"])
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# persistent cache DB (file lives alongside app unless absolute path provided)
GEOCODE_CACHE_DB = os.getenv("GEOCODE_CACHE_DB", "geocode_cache.sqlite")
GEOCODE_CACHE_TTL = int(os.getenv("GEOCODE_CACHE_TTL_SECS", str(24 * 3600)))
GEOCODE_MAX_QPS = int(os.getenv("GEOCODE_MAX_QPS", "5"))

# audit log path
AUDIT_LOG = os.getenv("AUDIT_LOG", "audit_log.csv")

# ---------- HTTP SESSION (reused + retries) ----------
def _build_session():
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

/* Chat bubbles */
.chat-bubble-bot {
  background: #e9eef9; color: #111827;
  border-left: 6px solid #3b82f6;
  border-radius: 8px; padding: 10px; margin-bottom: 8px; margin-right: 30%;
}
.chat-bubble-user {
  background: #3b82f6; color: #fff;
  border-radius: 8px; padding: 10px; margin-bottom: 8px; margin-left: 30%; text-align: right;
}

/* Slightly smaller grid font to reduce horizontal scrolling */
[data-testid="stDataEditor"] table { font-size: 0.94rem; }
</style>
""", unsafe_allow_html=True)

# ---------- ADDRESS ERROR DETECTION (configurable from .env) ----------
def _env_patterns(key: str, defaults: list[str]) -> list[str]:
    raw = (os.getenv(key, "") or "").strip()
    if not raw:
        return defaults
    return [p.strip() for p in raw.split(";") if p.strip()]

_DEFAULT_ADDRESS_ERROR_PATTERNS = [
    r"\binvalid\s*(postal\s*code|zip|zip\s*code|zipcode)\b",
    r"\b(postal\s*code|zip\s*code)\s*invalid\b",
    r"\baddress\s*1\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\baddress1\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\baddress(?:\s*line)?\s*1\s*(?:is\s*)?(?:empty|missing)\b",
    r"\borigin\s*country\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\bdestination\s*country\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\b(city|state|province|country)\s*(?:is\s*)?(invalid|empty|missing)\b",
]
_DEFAULT_NEGATIVE_PATTERNS = [
    r"\bhas\s+only\s+special\s+characters\s+in\s+item\b",
]

ADDRESS_ERROR_PATTERNS = _env_patterns("ADDR_PATTERNS", _DEFAULT_ADDRESS_ERROR_PATTERNS)
NEGATIVE_PATTERNS      = _env_patterns("NEG_ADDR_PATTERNS", _DEFAULT_NEGATIVE_PATTERNS)

def is_address_error(text: str) -> bool:
    if not text:
        return False
    for pat in NEGATIVE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return False
    for pat in ADDRESS_ERROR_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            return True
    return False

# ---------- CACHE HELPERS ----------
def normalize_address_for_cache(s: str) -> str:
    """Normalize a full-address string and hash it for stable in-memory keys (used in dedupe map)."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------- GEOCODING (with persistent cache + QPS limit) ----------
def correct_address(address: str):
    """
    Returns dict:
      - {"formatted_address": "...", "components": [...]}
      - or {"error_description_specific": "..."} / {"api_error": "..."}
    Persistent cache is checked first; on miss we call Google and cache the result.
    """
    if not GOOGLE_MAPS_API_KEY:
        return {"api_error": "Google Maps API key is missing. Set GOOGLE_MAPS_API_KEY in .env"}

    addr = (address or "").strip()
    if not addr or re.fullmatch(r"[\W_]+", addr):
        return {"error_description_specific": "Address input is empty or malformed."}

    # 1) persistent cache
    hit = cache_get(GEOCODE_CACHE_DB, addr, ttl_secs=GEOCODE_CACHE_TTL)
    if hit is not None:
        return hit

    # 2) external call (rate-limited)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={addr}&key={GOOGLE_MAPS_API_KEY}"
    try:
        respect_qps(GEOCODE_MAX_QPS)
        resp = HTTP.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        results = data.get("results", [])
        if status == "OK" and results:
            top = results[0]
            result = {
                "formatted_address": top.get("formatted_address", ""),
                "components": top.get("address_components", [])
            }
            cache_set(GEOCODE_CACHE_DB, addr, result)
            return result
        if status == "ZERO_RESULTS":
            result = {"error_description_specific": "No valid address found for the given input."}
            cache_set(GEOCODE_CACHE_DB, addr, result)
            return result
        result = {"error_description_specific":
                  f"Google Maps API error: {status or 'UNKNOWN_STATUS'} - {data.get('error_message', 'No message')}"}
        cache_set(GEOCODE_CACHE_DB, addr, result)
        return result
    except requests.exceptions.Timeout:
        return {"error_description_specific": "Address correction request timed out."}
    except requests.exceptions.RequestException as e:
        return {"error_description_specific": f"Network or API request error: {e}"}
    except Exception as e:
        return {"error_description_specific": f"Address correction processing error: {e}"}

def parse_address_components(components):
    parts = {
        "street_number": "", "route": "", "subpremise": "", "premise": "",
        "locality": "", "administrative_area_level_1": "", "postal_code": "",
        "country": "", "country_code": ""
    }
    for c in components:
        types = c.get("types", [])
        if "street_number" in types:
            parts["street_number"] = c.get("long_name", "")
        elif "route" in types:
            parts["route"] = c.get("long_name", "")
        elif "subpremise" in types:
            parts["subpremise"] = c.get("long_name", "")
        elif "premise" in types:
            parts["premise"] = c.get("long_name", "")
        elif "locality" in types:
            parts["locality"] = c.get("long_name", "")
        elif "administrative_area_level_1" in types:
            parts["administrative_area_level_1"] = c.get("short_name", "")
        elif "postal_code" in types:
            parts["postal_code"] = c.get("long_name", "")
        elif "country" in types:
            parts["country"] = c.get("long_name", "")
            parts["country_code"] = c.get("short_name", "")

    dest_address1 = " ".join(filter(None, [parts["street_number"], parts["route"]])).strip()
    return {
        "Destination Address 1": dest_address1,
        "Destination Address 2": parts["subpremise"].strip(),
        "Destination Address 3": parts["premise"].strip(),
        "Destination City": parts["locality"],
        "Destination State": parts["administrative_area_level_1"],
        "Destination ZIP": parts["postal_code"],
        "Destination Country": parts["country"],
        "Destination Country Code": parts["country_code"],
    }

# ---------- Optional LLM (intent fallback) ----------
def ollama_chat(prompt: str) -> str:
    try:
        r = HTTP.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=30
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        return f"[LLM error: {e}]"

# ---------- INTENT PARSER (lenient) ----------
ADDR_WORDS = [
    "address", "destination address", "city", "state", "zip", "postal", "postcode",
    "normalize address", "fix address", "clean address", "correct address", "change address",
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
    m = re.search(r"(?:correct|fix|update|change)\s+the\s+address\s+for\s+(.+)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def parse_user_intent(user_input: str, item_names):
    """
    Returns (intent, matched_item, address_string)
    Intents: batch_correct_addresses | correct_address_for_item | correct_single_address
    """
    text = (user_input or "").strip()
    if not text:
        return None, None, None

    matched_item = _extract_item_name(text, item_names)
    address_string = _extract_address_string(text)

    # Rule-based (fast & deterministic)
    if _contains_any(text, ADDR_WORDS) and _contains_any(text, BATCH_WORDS):
        return "batch_correct_addresses", None, None
    if _contains_any(text, ADDR_WORDS) and matched_item and any(v in text.lower() for v in SINGLE_VERBS):
        return "correct_address_for_item", matched_item, None
    if address_string:
        return "correct_single_address", None, address_string

    # LLM fallback (optional)
    if not USE_LOCAL_LLM:
        return None, None, None

    prompt = f"""
You classify a user's message into exactly one of:
- batch_correct_addresses
- correct_address_for_item
- correct_single_address

If user asks to fix addresses for every row -> batch_correct_addresses.
If they mention a specific item name -> correct_address_for_item.
If they provide a free-form address string ("correct the address for 123 Main...") -> correct_single_address.

Return JSON ONLY with keys: intent, item_name, address_string (null when missing).
Known item names: {list(item_names)}
User: \"\"\"{text}\"\"\"
JSON:
"""
    raw = ollama_chat(prompt)
    if raw.startswith("[LLM error:"):
        return None, None, None
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    blob = m.group(0) if m else raw
    try:
        import json
        data = json.loads(blob)
        intent = data.get("intent")
        item = data.get("item_name") or matched_item
        addr = data.get("address_string")
        if intent not in {"batch_correct_addresses", "correct_address_for_item", "correct_single_address"}:
            return None, None, None
        if intent == "correct_address_for_item" and not item:
            item = _extract_item_name(text, item_names)
        return intent, item, addr
    except Exception:
        return None, None, None

# ---------- SMALL UI HELPERS ----------
def chat_bubble(sender, message):
    if sender == "bot":
        st.markdown(f'<div class="chat-bubble-bot">{html.escape(message)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-user">{html.escape(message)}</div>', unsafe_allow_html=True)

def audit_rows(applied_props: list[dict], df: pd.DataFrame):
    """Append applied changes to AUDIT_LOG CSV."""
    if not applied_props:
        return
    path = AUDIT_LOG
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["ts","user","row_idx","item","field","old","new"])
        ts = datetime.utcnow().isoformat()
        user = st.session_state.get("username", "unknown")
        for p in applied_props:
            idx = p["idx"]; item = p.get("item_name","")
            comps = p.get("new_components", {})
            for field in ["Destination Address 1","Destination Address 2","Destination Address 3",
                          "Destination City","Destination State","Destination ZIP","Destination Country","Destination Country Code"]:
                if field in df.columns:
                    old = df.at[idx, field]
                    new = comps.get(field, "")
                    if str(old) != str(new):
                        w.writerow([ts, user, idx, item, field, old, new])

# ---------- STREAMLIT APP ----------
async def async_main():
    st.title("Corrective AI — Address Correction")

    # Session state
    ss = st.session_state
    if "results_df" not in ss:
        ss.results_df = pd.DataFrame()
    if "chat_history" not in ss:
        ss.chat_history = []
    if "proposals" not in ss:
        ss.proposals = []  # list[dict]
    if "proposals_df" not in ss:
        ss.proposals_df = pd.DataFrame()
    if "csv_loaded" not in ss:
        ss.csv_loaded = False
    if "review_msg" not in ss:
        ss.review_msg = ""

    # ----------------- Sidebar: Actions -----------------
    with st.sidebar:
        st.header("Actions")
        uploaded = st.file_uploader(
            "Upload CSV (address columns + Error Description)",
            type=["csv"],
            help="Required: Item Name, Destination Address 1/2/3, City, State, ZIP, Country, Country Code, Error Description"
        )

        with st.form("cmd_form", clear_on_submit=True):
            user_input = st.text_input(
                "Command (lenient)",
                placeholder="e.g., fix all addresses / change the address for Blue Wallet / correct the address for 123 Main St..."
            )
            submitted = st.form_submit_button("Send")

        if not ss.results_df.empty:
            st.download_button(
                "💾 Download Current CSV",
                data=ss.results_df.to_csv(index=False).encode("utf-8"),
                file_name="corrected_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        if st.button("Clear cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Streamlit cache cleared (persistent cache is separate).")

        if st.button("Reset", use_container_width=True):
            ss.results_df = pd.DataFrame()
            ss.chat_history = []
            ss.proposals = []
            ss.proposals_df = pd.DataFrame()
            ss.csv_loaded = False
            ss.review_msg = ""
            st.rerun()

        st.markdown("**Examples:**")
        st.markdown("- fix all addresses\n- change the address for **Blue Wallet**\n- correct the address for **123 Main St, Springfield, IL**")

    # ----------------- Main: Tabs -----------------
    tab_proposals, tab_results, tab_chat = st.tabs(["Proposals", "Results", "Chat"])

    with tab_results:
        if not ss.results_df.empty:
            st.dataframe(ss.results_df, use_container_width=True)

    with tab_chat:
        for sender, msg in ss.chat_history:
            chat_bubble(sender, msg)

    with tab_proposals:
        if not ss.proposals_df.empty:
            st.markdown("### Proposed Address Corrections")
            if ss.review_msg:
                st.info(ss.review_msg)

            # Stable data_editor
            col_order = [
                "Approve",
                "Row Index", "Item Name",
                "Original Full Address", "Proposed Full Address",
                "Proposed Address 1", "Proposed Address 2", "Proposed Address 3",
                "Proposed City", "Proposed State", "Proposed ZIP",
                "Proposed Country", "Proposed Country Code",
            ]
            col_order = [c for c in col_order if c in ss.proposals_df.columns]

            column_config = {
                "Approve": st.column_config.CheckboxColumn(
                    "Approve", help="Tick to accept this row", default=False, width="small"
                ),
                "Row Index": st.column_config.NumberColumn("Row", width="small"),
                "Item Name": st.column_config.TextColumn("Item", width="medium", max_chars=24),
                "Original Full Address": st.column_config.TextColumn(
                    "Original Full Address", width="large", max_chars=64
                ),
                "Proposed Full Address": st.column_config.TextColumn(
                    "Proposed Full Address", width="large", max_chars=64
                ),
                "Proposed Address 1": st.column_config.TextColumn("Addr 1", width="medium", max_chars=28),
                "Proposed Address 2": st.column_config.TextColumn("Addr 2", width="small", max_chars=18),
                "Proposed Address 3": st.column_config.TextColumn("Addr 3", width="small", max_chars=18),
                "Proposed City":   st.column_config.TextColumn("City",   width="small", max_chars=18),
                "Proposed State":  st.column_config.TextColumn("State",  width="small", max_chars=10),
                "Proposed ZIP":    st.column_config.TextColumn("ZIP",    width="small", max_chars=10),
                "Proposed Country":     st.column_config.TextColumn("Country", width="small", max_chars=12),
                "Proposed Country Code":st.column_config.TextColumn("CC",      width="small", max_chars=4),
            }

            table_height = min(600, 140 + 32 * len(ss.proposals_df))

            edited_df = st.data_editor(
                ss.proposals_df,
                key="proposals_editor",
                use_container_width=True,
                num_rows="fixed",
                height=table_height,
                hide_index=True,
                column_order=col_order,
                column_config=column_config,
                disabled=[
                    "Row Index", "Item Name",
                    "Original Full Address", "Proposed Full Address",
                    "Proposed Address 1", "Proposed Address 2", "Proposed Address 3",
                    "Proposed City", "Proposed State", "Proposed ZIP",
                    "Proposed Country", "Proposed Country Code",
                ],
            )
            st.session_state.proposals_df = edited_df

            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("✅ Apply Selected", key="apply_selected", type="primary", use_container_width=True):
                    df = ss.results_df
                    selected_rows = edited_df[edited_df.get("Approve", False) == True]
                    selected_idxs = set(int(x) for x in selected_rows["Row Index"].tolist())

                    # Build quick lookup for proposals by idx
                    pmap = {p["idx"]: p for p in ss.proposals}
                    applied_props = []

                    for idx in selected_idxs:
                        p = pmap.get(idx)
                        if not p:
                            continue
                        comps = p.get("new_components", {})
                        for col, val in comps.items():
                            if col in df.columns:
                                df.at[idx, col] = val
                        if "Error" in df.columns:
                            df.at[idx, "Error"] = ""
                        if "Error Description" in df.columns:
                            df.at[idx, "Error Description"] = ""
                        applied_props.append(p)

                    # audit
                    audit_rows(applied_props, df)

                    # Remove applied from proposals
                    remaining = [p for p in ss.proposals if p["idx"] not in selected_idxs]
                    ss.proposals = remaining
                    if remaining:
                        rem_df = edited_df[~edited_df["Row Index"].isin(selected_idxs)].copy()
                        rem_df["Approve"] = False
                        ss.proposals_df = rem_df
                    else:
                        ss.proposals_df = pd.DataFrame()

                    ss.results_df = df
                    ss.review_msg = f"Applied {len(applied_props)} selected row(s)."
                    st.rerun()

            with a2:
                if st.button("✅ Accept All Changes", key="accept_all", use_container_width=True):
                    df = ss.results_df
                    applied_props = []
                    for p in ss.proposals:
                        idx = p["idx"]
                        comps = p.get("new_components", {})
                        for col, val in comps.items():
                            if col in df.columns:
                                df.at[idx, col] = val
                        if "Error" in df.columns:
                            df.at[idx, "Error"] = ""
                        if "Error Description" in df.columns:
                            df.at[idx, "Error Description"] = ""
                        applied_props.append(p)

                    # audit
                    audit_rows(applied_props, df)

                    ss.results_df = df
                    ss.proposals = []
                    ss.proposals_df = pd.DataFrame()
                    ss.review_msg = "Applied all proposed address corrections."
                    st.rerun()

            with a3:
                if st.button("❌ Reject All Changes", key="reject_all", use_container_width=True):
                    ss.proposals = []
                    ss.proposals_df = pd.DataFrame()
                    ss.review_msg = "Discarded all proposed corrections."
                    st.rerun()
        else:
            st.info("No proposed corrections yet. Run a command like “fix all addresses”.")

    # ----------------- Handle CSV upload -----------------
    if 'uploaded' in locals() and uploaded and not ss.csv_loaded:
        try:
            dtype_map = {
                'Item Name': str,
                'Destination Address 1': str,
                'Destination Address 2': str,
                'Destination Address 3': str,
                'Destination City': str,
                'Destination State': str,
                'Destination ZIP': str,
                'Destination Country': str,
                'Destination Country Code': str,
                'Error Description': str,
                'Error': str
            }
            df = pd.read_csv(uploaded, dtype=dtype_map, keep_default_na=False).fillna('')
            required = [
                'Item Name',
                'Destination Address 1', 'Destination Address 2', 'Destination Address 3',
                'Destination City', 'Destination State', 'Destination ZIP',
                'Destination Country', 'Destination Country Code',
                'Error Description'
            ]
            for col in required:
                if col not in df.columns:
                    df[col] = ""
            if 'Error' not in df.columns:
                df['Error'] = ""

            ss.results_df = df
            ss.csv_loaded = True
            ss.chat_history.append(("bot", "CSV uploaded. You can now say things like 'fix all addresses' or 'change the address for Blue Wallet'."))
            st.rerun()
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

    # ----------------- Handle commands -----------------
    if 'submitted' in locals() and submitted and (user_input or "").strip():
        ss.chat_history.append(("user", user_input))
        df = ss.results_df

        if df.empty:
            ss.chat_history.append(("bot", "Please upload a CSV first."))
            st.rerun()

        item_names = list(df["Item Name"].dropna().unique()) if "Item Name" in df.columns else []
        intent, matched_item, address_string = parse_user_intent(user_input, item_names)

        # ---- Batch address correction ----
        if intent == "batch_correct_addresses":
            ss.proposals = []
            ss.proposals_df = pd.DataFrame()
            with st.spinner("Correcting addresses in batch..."):
                addr_map = {}
                for idx, row in df.iterrows():
                    err_desc = (row.get("Error Description", "") or "").strip()
                    if not is_address_error(err_desc):
                        continue
                    parts = [
                        row.get("Destination Address 1", "").strip(),
                        row.get("Destination Address 2", "").strip(),
                        row.get("Destination Address 3", "").strip(),
                        row.get("Destination City", "").strip(),
                        row.get("Destination State", "").strip(),
                        row.get("Destination ZIP", "").strip(),
                        row.get("Destination Country", "").strip(),
                    ]
                    original_full = ", ".join([p for p in parts if p])
                    if not original_full or re.fullmatch(r"[\W_]+", original_full):
                        continue
                    key = normalize_address_for_cache(original_full)
                    addr_map.setdefault(key, {"full": original_full, "rows": []})["rows"].append(idx)

                display_rows = []
                for key, bundle in addr_map.items():
                    corr = correct_address(bundle["full"])
                    if "formatted_address" not in corr:
                        continue
                    comps = parse_address_components(corr["components"])
                    if not comps.get("Destination ZIP", "").strip():
                        continue
                    for idx in bundle["rows"]:
                        row = df.loc[idx]
                        ss.proposals.append({
                            "idx": idx,
                            "item_name": row.get("Item Name", f"Row {idx+1}"),
                            "original_full": bundle["full"],
                            "formatted_new_address": corr["formatted_address"],
                            "new_components": comps
                        })
                        display_rows.append({
                            "Row Index": idx,
                            "Item Name": row.get("Item Name", f"Row {idx+1}"),
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
            ss.chat_history.append(("bot", ss.review_msg))
            st.rerun()

        # ---- Single address (by item name) ----
        elif intent == "correct_address_for_item" and matched_item:
            rows = df.index[df["Item Name"] == matched_item].tolist()
            if not rows:
                ss.chat_history.append(("bot", f"Could not find an item named '{matched_item}'."))
                st.rerun()
            idx = rows[0]
            row = df.loc[idx]

            err_desc = (row.get("Error Description", "") or "").strip()
            if not is_address_error(err_desc):
                ss.chat_history.append(("bot", f"No address-related issue found in Error Description for '{matched_item}'."))
                st.rerun()

            parts = [
                row.get("Destination Address 1", "").strip(),
                row.get("Destination Address 2", "").strip(),
                row.get("Destination Address 3", "").strip(),
                row.get("Destination City", "").strip(),
                row.get("Destination State", "").strip(),
                row.get("Destination ZIP", "").strip(),
                row.get("Destination Country", "").strip(),
            ]
            original_full = ", ".join([p for p in parts if p])
            if not original_full or re.fullmatch(r"[\W_]+", original_full):
                ss.chat_history.append(("bot", f"No valid original address found for '{matched_item}' to correct."))
                st.rerun()

            with st.spinner(f"Correcting address for '{matched_item}'..."):
                corr = correct_address(original_full)
                if "formatted_address" in corr:
                    comps = parse_address_components(corr["components"])
                    if comps.get("Destination ZIP", "").strip():
                        ss.proposals = [{
                            "idx": idx,
                            "item_name": matched_item,
                            "original_full": original_full,
                            "formatted_new_address": corr["formatted_address"],
                            "new_components": comps
                        }]
                        ss.proposals_df = pd.DataFrame([{
                            "Row Index": idx,
                            "Item Name": matched_item,
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
                        ss.review_msg = f"Proposed correction for '{matched_item}'."
                    else:
                        ss.chat_history.append(("bot", "Corrected address has no ZIP; not proposed."))
                else:
                    err = corr.get("api_error") or corr.get("error_description_specific")
                    ss.chat_history.append(("bot", f"Address correction failed: {err}"))
            st.rerun()

        # ---- Free-form single address string ----
        elif intent == "correct_single_address" and address_string:
            with st.spinner(f"Correcting: {address_string}"):
                corr = correct_address(address_string)
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
            else:
                err = corr.get("api_error") or corr.get("error_description_specific")
                ss.chat_history.append(("bot", f"Could not correct: {err}"))
            st.rerun()

        else:
            ss.chat_history.append(("bot", "I didn't catch that. Try: 'fix all addresses' or 'change the address for Blue Wallet'."))
            st.rerun()

    # Initial greeting
    if not ss.csv_loaded and not ss.chat_history:
        ss.chat_history.append(("bot", "Hello! Upload a CSV to get started."))
        st.rerun()


if __name__ == "__main__":
    asyncio.run(async_main())
