import os

APP_NAME = "life_insurance_advisory"
DEFAULT_USER_ID = "streamlit_user"

# Override in backend/.env with GEMINI_MODEL=... if needed.
# gemini-2.5-pro is used by default (flash models often hit 503 during demand spikes).
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Tried in order when the API returns temporary overload errors (503/429).
FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash-lite,gemini-2.5-flash",
    ).split(",")
    if model.strip()
]
