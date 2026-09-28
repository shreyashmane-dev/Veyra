"""Root entry point for running the ZONARETH application.

Run with:
    python app.py
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from zonareth.app import main
except ImportError:
    from veyra.app import main

if __name__ == "__main__":
    main()
