"""aerotrack: drone-video detection + tracking under an embedded latency budget."""

import os
from pathlib import Path

# Must run before anything imports ultralytics:
# - a repo-local settings dir, so runs never depend on a user-global settings.json;
# - no silent `pip install` at runtime, which would bypass the uv lockfile.
_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(_REPO_ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

__version__ = "0.1.0"
