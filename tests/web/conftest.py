import sys
from pathlib import Path

# Reuse the project conftest path setup (src + 04_coaching already on path via
# tests/conftest.py). Add web/backend so `import readers`, `import settings` work.
_WEB = Path(__file__).resolve().parent.parent.parent / "web" / "backend"
sys.path.insert(0, str(_WEB))