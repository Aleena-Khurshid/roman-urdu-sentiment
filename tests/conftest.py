import os
import sys
from pathlib import Path

# Make "app" and "src" importable, and skip loading the real model in tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PRELOAD_MODEL"] = "0"
