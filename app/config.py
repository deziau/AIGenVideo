import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("AIGV_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"

# Longest side of processed video, in pixels. Keeps local stylizing fast.
MAX_SIDE = int(os.environ.get("AIGV_MAX_SIDE", "1280"))
# Seconds of the reference clip sent to the generative model.
REFERENCE_SECONDS = float(os.environ.get("AIGV_REFERENCE_SECONDS", "5"))
MAX_UPLOAD_MB = int(os.environ.get("AIGV_MAX_UPLOAD_MB", "200"))

CLAUDE_MODEL = os.environ.get("AIGV_CLAUDE_MODEL", "claude-opus-5")
RUNWAY_API_BASE = os.environ.get("RUNWAY_API_BASE", "https://api.dev.runwayml.com/v1")
RUNWAY_API_VERSION = "2024-11-06"


def anthropic_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def runway_enabled() -> bool:
    return bool(os.environ.get("RUNWAYML_API_SECRET"))


for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
