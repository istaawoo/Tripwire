import os
from dotenv import load_dotenv

load_dotenv()


def _get(name, default=""):
    # strip inline "# comment" that .env.example uses
    return os.getenv(name, default).split("#")[0].strip()


GENERATOR_BASE_URL = _get("GENERATOR_BASE_URL", "http://localhost:11434/v1")
GENERATOR_MODEL = _get("GENERATOR_MODEL", "gemma4")
GENERATOR_API_KEY = _get("GENERATOR_API_KEY", "ollama")

STAGE1_BASE_URL = _get("STAGE1_BASE_URL", "http://localhost:11434/v1")
STAGE1_MODEL = _get("STAGE1_MODEL", "REPLACE_ME")
STAGE1_API_KEY = _get("STAGE1_API_KEY", "ollama")

STAGE2_BASE_URL = _get("STAGE2_BASE_URL", "http://localhost:11434/v1")
STAGE2_MODEL = _get("STAGE2_MODEL", "REPLACE_ME")
STAGE2_API_KEY = _get("STAGE2_API_KEY", "ollama")

STAGE1_THRESHOLD = float(_get("STAGE1_THRESHOLD", "0.3"))

ROLES = {
    "generator": (GENERATOR_BASE_URL, GENERATOR_MODEL, GENERATOR_API_KEY),
    "stage1": (STAGE1_BASE_URL, STAGE1_MODEL, STAGE1_API_KEY),
    "stage2": (STAGE2_BASE_URL, STAGE2_MODEL, STAGE2_API_KEY),
}
