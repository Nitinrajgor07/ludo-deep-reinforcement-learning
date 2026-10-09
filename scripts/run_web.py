"""Launcher script for PhD Ludo DRL FastAPI Web Application.

Runs:
    python -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
"""

import sys
import uvicorn


def main():
    print("=" * 70)
    print("  Starting PhD Ludo DRL Web Application (FastAPI + Uvicorn)")
    print("  Host: http://127.0.0.1:8000 / http://localhost:8000")
    print("  Theme: 100% Light Glassmorphism")
    print("=" * 70)
    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
