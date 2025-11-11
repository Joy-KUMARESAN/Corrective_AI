
UNWANTED_PATTERNS = [
    r'error\s*=\s*None',
    r'include_in_memory\s*=\s*False',
    r'all_model_outputs\s*=\s*\[.*',
    r'DOMHistoryElement\(',
    r'\],\s*all_model_outputs',
    r'\{.*\}',
    r'\[.*\]',
    r'Playwright not supported',
    r'404 error',
    r'^\s*$',
    r"success['\"]?\s*:\s*True",
    r"interacted_element['\"]?\s*:\s*None",
    r"done['\"]?\s*:\s*\{.*",
    r"agent\s*:",
    r"extract_content\s*:",
    r"\),",
]

# --------------------------------------------------------------------
# Address-related error detectors
# These patterns are intentionally broad so that address correction
# runs whenever *any* address signal appears in the Error Description.
# Matching is case-insensitive (handled in logic.address.is_address_error).
# --------------------------------------------------------------------
ADDRESS_ERROR_PATTERNS = [
    # Explicit “invalid postal” / zip / postcode
    r'\binvalid\s*(postal\s*code|postcode|zip)\b',
    r'\b(zip|postal|postcode)\s*(code)?\s*is\s*(invalid|empty|missing)\b',

    # Mentions of postal/zip/postcode anywhere (e.g. “…;Invalid Postal Code”)
    r'\b(postal|postcode|zip)\b',

    # Missing/empty core address parts
    r'\b(missing|empty)\s*(address|address\s*line\s*1|address\s*line\s*2|street|route|city|state|province|zip|postal|postcode|country)\b',

    # Generic address cues
    r'\b(address|street|route|road|premise|subpremise|locality|city|state|province|country)\s*(is)?\s*(invalid|missing|empty)\b',
    r'\bincomplete\s*address\b',
    r'\bno\s*valid\s*address\b',
    r'\bincorrect\s*(address|city|state|zip|postal|postcode|country)\b',

    # Origin/Destination country problems (covering slashes and punctuation)
    r'\borigin\s*country\b.*\b(empty|missing|invalid|only\s*special\s*characters)\b',
    r'\borigincountry\b\s*(is)?\s*(empty|missing)\b',
    r'\borigincountry\b.*\bonly\s*special\s*characters\b',
    r'\bdestination\s*country\b.*\b(empty|missing|invalid)\b',
]

# --------------------------------------------------------------------
# Exclusions — prevent false positives that should NOT trigger address correction
# --------------------------------------------------------------------
ADDRESS_ERROR_EXCLUDE_PATTERNS = [
    # This is NOT an address error per your requirement:
    r'has\s+only\s+special\s+characters\s+in\s+item',
]
