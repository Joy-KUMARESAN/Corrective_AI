# src/corrective_ai/logic/address.py
import re
from typing import Dict, List


def parse_address_components(components: List[Dict]) -> Dict[str, str]:
    """
    Parses Google Maps API address components into a dictionary
    suitable for updating DataFrame columns, including country code and premise.
    """
    address_parts = {
        "street_number": "",
        "route": "",
        "subpremise": "",
        "premise": "",
        "locality": "",
        "administrative_area_level_1": "",
        "postal_code": "",
        "country": "",
        "country_code": ""
    }
    for component in components:
        types = component.get("types", [])
        if 'street_number' in types:
            address_parts['street_number'] = component.get('long_name', '')
        elif 'route' in types:
            address_parts['route'] = component.get('long_name', '')
        elif 'subpremise' in types:
            address_parts['subpremise'] = component.get('long_name', '')
        elif 'premise' in types:
            address_parts['premise'] = component.get('long_name', '')
        elif 'locality' in types:
            address_parts['locality'] = component.get('long_name', '')
        elif 'administrative_area_level_1' in types:
            # short_name keeps US state abbreviations like "CA"
            address_parts['administrative_area_level_1'] = component.get('short_name', '')
        elif 'postal_code' in types:
            address_parts['postal_code'] = component.get('long_name', '')
        elif 'country' in types:
            address_parts['country'] = component.get('long_name', '')
            address_parts['country_code'] = component.get('short_name', '')

    dest_address1_parts = [address_parts['street_number'], address_parts['route']]
    dest_address1 = " ".join(filter(None, dest_address1_parts)).strip()
    dest_address2 = address_parts['subpremise'].strip()
    dest_address3 = address_parts['premise'].strip()

    return {
        "Destination Address 1": dest_address1,
        "Destination Address 2": dest_address2,
        "Destination Address 3": dest_address3,
        "Destination City": address_parts['locality'],
        "Destination State": address_parts['administrative_area_level_1'],
        "Destination ZIP": address_parts['postal_code'],
        "Destination Country Code": address_parts['country_code'],
        "Destination Country": address_parts['country']
    }


# --------------------------------------------------------------------
# Address-error detection
# --------------------------------------------------------------------
# We treat these as address-related error phrases. The patterns are case-insensitive.
ADDRESS_ERROR_PATTERNS = [
    # Invalid ZIP / Postal Code
    r"\binvalid\s*(postal\s*code|zip|zip\s*code|zipcode)\b",
    r"\b(postal\s*code|zip\s*code)\s*invalid\b",

    # Address1 / Address line 1 empty / only special characters
    r"\baddress\s*1\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\baddress1\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\baddress(?:\s*line)?\s*1\s*(?:is\s*)?(?:empty|missing)\b",

    # Destination address generics
    r"\bdestination\s*address\b.*\b(invalid|empty|missing)\b",

    # Origin/Destination country empty / only special chars
    r"\borigin\s*country\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",
    r"\bdestination\s*country\s*is\s*empty\s*/\s*has\s*only\s*special\s*characters\b",

    # City/State/Province/Country invalid or empty (keep moderate, not too broad)
    r"\b(city|state|province|country)\s*(?:is\s*)?(invalid|empty|missing)\b",
]

# Phrases we explicitly DO NOT treat as address errors
NEGATIVE_PATTERNS = [
    r"\bhas\s+only\s+special\s+characters\s+in\s+item\b",  # not address-related
]


def is_address_error(text: str) -> bool:
    """
    Returns True if 'text' looks like an address-related problem.
    This centralizes all triggers used throughout the app.
    """
    if not text:
        return False

    # If any negative pattern is present, short-circuit to False
    for pat in NEGATIVE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return False

    # Otherwise, see if any of our address patterns match
    for pat in ADDRESS_ERROR_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            return True

    return False
