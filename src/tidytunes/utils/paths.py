import os
from pathlib import Path

ROOT = Path(os.environ.get("GLOBAL_ROOT", Path(__file__).parent))
ARTIFACTS = ROOT / "artifacts"
