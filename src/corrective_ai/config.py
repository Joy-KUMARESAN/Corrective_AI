# src/corrective_ai/config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    HS_CODE_CSV: str
    GOOGLE_MAPS_API_KEY: str
    USE_LOCAL_LLM: bool = True
    LLM_MODEL_NAME: str = "mistral"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

def get_config() -> Config:
    return Config(
        HS_CODE_CSV=os.getenv("HS_CODE_CSV", ""),
        GOOGLE_MAPS_API_KEY=os.getenv("GOOGLE_MAPS_API_KEY", ""),
        USE_LOCAL_LLM=(os.getenv("USE_LOCAL_LLM", "true").lower() in ["1", "true", "yes"]),
        LLM_MODEL_NAME=os.getenv("LLM_MODEL_NAME", "mistral"),
        OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
