import requests
from typing import Dict, Any
from ..config import get_config

def correct_address(address: str) -> Dict[str, Any]:
    """
    Corrects and standardizes an address using the Google Maps Geocoding API.
    Returns a dict with either:
      - {"formatted_address": ..., "components": [...]} on success
      - {"error_description_specific": "..."} on API/processing issues
      - {"api_error": "..."} if API key missing
    """
    cfg = get_config()
    api_key = cfg.GOOGLE_MAPS_API_KEY

    if not api_key:
        return {"api_error": "Google Maps API key is missing. Please set GOOGLE_MAPS_API_KEY in your .env file."}

    addr = (address or "").strip()
    if not addr:
        return {"error_description_specific": "Address input is empty or null."}

    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={addr}&key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        results = data.get("results", [])
        if status == "OK" and results:
            top = results[0]
            return {
                "formatted_address": top.get("formatted_address", ""),
                "components": top.get("address_components", []),
            }
        elif status == "ZERO_RESULTS":
            return {"error_description_specific": "No valid address found for the given input."}
        else:
            return {
                "error_description_specific": f"Google Maps API error: {status or 'UNKNOWN_STATUS'} - {data.get('error_message', 'No specific error message.')}"
            }

    except requests.exceptions.Timeout:
        return {"error_description_specific": "Address correction request timed out."}
    except requests.exceptions.RequestException as e:
        return {"error_description_specific": f"Network or API request error: {e}"}
    except Exception as e:
        return {"error_description_specific": f"Address correction processing error: {e}"}
