from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
PERSONAL = DATA / "personal"
CLIPS = PERSONAL / "clips"
MANIFEST = PERSONAL / "manifest.jsonl"
PERSONAL_WEIGHTS = PERSONAL / "model_personal.pth"
PERSONAL_DECODE = PERSONAL / "decode.json"
PERSONAL_CONTEXT = PERSONAL / "context.txt"
REPORTS = PERSONAL / "reports"
HISTORY = DATA / "history"
SETTINGS = DATA / "settings.json"
APP_LOG = DATA / "app.log"
APP_LOCK = DATA / "app.lock"

VSR_DIR = MODELS / "vsr_lrs3"
LM_DIR = MODELS / "lm_lrs3"
SPM_MODEL = MODELS / "unigram5000.model"
SPM_UNITS = MODELS / "unigram5000_units.txt"
FACE_MODEL = MODELS / "mediapipe" / "face_landmarker.task"
