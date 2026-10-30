import os
import sys
import tempfile
from pathlib import Path

# Personal data goes to a throwaway folder; must be set before silentdict.paths is imported.
os.environ["SILENTDICT_DATA"] = tempfile.mkdtemp(prefix="silentdict-test-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
