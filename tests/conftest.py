"""pytest configuration: ensure project root is on sys.path."""
import sys
from pathlib import Path

# Add the repository root (mso_blood_delivery/) to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
