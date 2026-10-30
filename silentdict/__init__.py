"""Silent Dictation: local, camera-only silent speech dictation for macOS."""
import sys
from pathlib import Path

__version__ = "0.2.0"

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

CREDITS = (
    "Inspired by Aarush Yadav's Subvocal announcement "
    "(https://x.com/AarushYadav47/status/2116198463847408066). "
    "This project is an independent camera-only experiment and is not affiliated with Subvocal."
)
