import json
import re
from typing import List, Tuple, Optional

from src.corrective_ai.config import get_config
from src.corrective_ai.services.llm import ollama_chat


ADDR_WORDS = [
    "address", "destination address", "city", "state", "zip", "postal", "postcode",
    "normalize address", "fix address", "clean address", "correct address", "change address",
]
HS_WORDS = [
    "hs code", "hscode", "harmonized", "tariff", "classification",
    "fix hs", "clean hs", "correct hs", "update hs",
]

BATCH_WORDS = [
    "all", "every", "entire file", "bulk", "batch", "everything", "all rows",
]

SINGLE_VERBS = [
    "fix", "correct", "change", "update", "modify", "edit", "set"
]

def _contains_any(text: str, words: List[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)

def _extract_item_name(text: str, item_names: List[str]) -> Optional[str]:
    t = text.lower()
    for name in sorted(item_names, key=len, reverse=True):
        if name and name.lower() in t:
            return name
    return None

def _extract_address_string(text: str) -> Optional[str]:
    m = re.search(r"(?:correct|fix|update|change)\s+the\s+address\s+for\s+(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def parse_user_intent(user_input: str, item_names: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (intent, matched_item, address_string).
    Intents:
      - batch_correct_addresses
      - correct_address_for_item
      - batch_correct_hs_codes
      - correct_single_hs_code_for_item
      - correct_single_address (free string)  [kept for backward compatibility]
    """
    text = (user_input or "").strip()
    if not text:
        return None, None, None

    matched_item = _extract_item_name(text, item_names)
    address_string = _extract_address_string(text)

    if _contains_any(text, ADDR_WORDS) and _contains_any(text, BATCH_WORDS):
        return "batch_correct_addresses", None, None

    if _contains_any(text, ADDR_WORDS) and matched_item and any(v in text.lower() for v in SINGLE_VERBS):
        return "correct_address_for_item", matched_item, None

    # Free-form address string (“correct the address for 123 Main…”)
    if address_string:
        return "correct_single_address", None, address_string

    # Batch HS
    if _contains_any(text, HS_WORDS) and _contains_any(text, BATCH_WORDS):
        return "batch_correct_hs_codes", None, None

    # Single HS by item
    if _contains_any(text, HS_WORDS) and matched_item and any(v in text.lower() for v in SINGLE_VERBS):
        return "correct_single_hs_code_for_item", matched_item, None

    # 2) If still unknown, try local LLM (Mistral via Ollama)
    cfg = get_config()
    if not cfg.USE_LOCAL_LLM:
        return None, None, None

    prompt = f"""
You are a classifier for an internal data-cleaning app. The user writes informal messages.
Pick ONE of these intents and extract fields:

Intents (exact strings):
- batch_correct_addresses
- correct_address_for_item
- batch_correct_hs_codes
- correct_single_hs_code_for_item
- correct_single_address   # when the user provides a free-form address string

Rules:
- If the user wants to update addresses for every row, choose batch_correct_addresses.
- If they mention an item name and want to fix JUST that item's address, choose correct_address_for_item.
- If they want to update HS codes for every row, choose batch_correct_hs_codes.
- If they mention an item name and want to fix JUST that item's HS code, choose correct_single_hs_code_for_item.
- If they say something like "correct the address for 123 Main St..." with no item name, choose correct_single_address and put the address string in "address_string".
- Return strictly JSON with keys: intent, item_name, address_string. Use null for missing values.

Known item names (choose exact match if you see them): {item_names}

User message:
\"\"\"{text}\"\"\" 

Output JSON only:
"""
    raw = ollama_chat(prompt, model=cfg.LLM_MODEL_NAME, timeout=30)

    # Basic guard: if the model returns an error string, just give up
    if raw.startswith("[LLM error:"):
        return None, None, None

    # Try to find the first JSON object in the text
    json_text = raw.strip()
    # Some models add prose; isolate a JSON block if present
    m = re.search(r"\{.*\}", json_text, flags=re.DOTALL)
    if m:
        json_text = m.group(0)

    try:
        data = json.loads(json_text)
        intent = data.get("intent")
        item = data.get("item_name") or matched_item  # prefer LLM, fallback to earlier match
        addr = data.get("address_string")

        # Validate intent
        valid = {
            "batch_correct_addresses",
            "correct_address_for_item",
            "batch_correct_hs_codes",
            "correct_single_hs_code_for_item",
            "correct_single_address",
        }
        if intent not in valid:
            return None, None, None

        # If LLM says correct_address_for_item but didn't supply item, try deterministic match
        if intent == "correct_address_for_item" and not item:
            item = _extract_item_name(text, item_names)

        # If LLM says correct_single_hs_code_for_item but didn't supply item, try deterministic match
        if intent == "correct_single_hs_code_for_item" and not item:
            item = _extract_item_name(text, item_names)

        return intent, item, addr
    except Exception:
        # If JSON parse fails, we give up and return None
        return None, None, None
