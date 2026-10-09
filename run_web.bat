@echo off
REM =========================================================================
REM  PhD Ludo DRL Platform — Local Web Server Launcher
REM  Starts FastAPI application with Uvicorn on http://127.0.0.1:8000
REM =========================================================================

echo Starting PhD Ludo DRL Web Application...
echo Target: http://127.0.0.1:8000
echo Target: http://localhost:8000
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
pause
