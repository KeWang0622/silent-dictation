"""Filesystem locations.

Code and models live next to this package (the source checkout, or the app's runtime copy).
Personal data (settings, calibration clips, personal model, history) lives in
~/Library/Application Support/Silent Dictation/data so the command line and the app share it.
Set SILENTDICT_DATA to use another folder (tests do this).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SUPPORT = Path.home() / "Library" / "Application Support" / "Silent Dictation"
DATA = Path(os.environ.get("SILENTDICT_DATA") or SUPPORT / "data")
PERSONAL = DATA / "personal"
CLIPS = PERSONAL / "clips"
MANIFEST = PERSONAL / "manifest.jsonl"
PERSONAL_WEIGHTS = PERSONAL / "model_personal.pth"
PERSONAL_DECODE = PERSONAL / "decode.json"
PERSONAL_FLARE_WEIGHTS = PERSONAL / "flare_personal.pth"
PERSONAL_FLARE_DECODE = PERSONAL / "flare_decode.json"
PERSONAL_CONTEXT = PERSONAL / "context.txt"
REPORTS = PERSONAL / "reports"
HISTORY = DATA / "history"
SETTINGS = DATA / "settings.json"
STATS = DATA / "stats.json"
APP_LOG = DATA / "app.log"
APP_LOCK = DATA / "app.lock"

VSR_DIR = MODELS / "vsr_lrs3"
LM_DIR = MODELS / "lm_lrs3"
SPM_MODEL = MODELS / "unigram5000.model"
SPM_UNITS = MODELS / "unigram5000_units.txt"
FACE_MODEL = MODELS / "mediapipe" / "face_landmarker.task"
LLM_DIR = MODELS / "llm" / "Qwen3-4B-Instruct-2507-4bit"
FLARE_WEIGHTS = MODELS / "flare_large" / "model.pt"
