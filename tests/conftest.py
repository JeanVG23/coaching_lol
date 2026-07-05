import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "core"))
sys.path.insert(0, str(_SRC / "collection"))
sys.path.insert(0, str(_SRC / "pipeline_ops"))
sys.path.insert(0, str(_SRC / "reporting"))
sys.path.insert(0, str(_SRC / "04_coaching"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # fixtures partagées entre tests
