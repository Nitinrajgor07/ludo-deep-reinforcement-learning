@echo off
REM =========================================================================
REM  PhD Ludo DRL Platform — LAN / Local Network Web Server Launcher
REM  Binds FastAPI/Uvicorn to 0.0.0.0:8000 for multi-device & multiplayer access
REM =========================================================================

echo =========================================================================
echo  Starting PhD Ludo DRL Web Application in LAN Mode (0.0.0.0:8000)...
echo =========================================================================
echo.
echo  Local PC Access:
echo    http://127.0.0.1:8000
echo    http://localhost:8000
echo.
echo  LAN / Wi-Fi Access for Other Computers, Tablets, or Phones:
echo    http://^<YOUR-LOCAL-IP^>:8000
echo.
echo  HOW TO FIND YOUR LOCAL IP:
echo    1. Open a new Command Prompt (cmd.exe)
echo    2. Run: ipconfig
echo    3. Look for "IPv4 Address" under your active Wi-Fi or Ethernet adapter
echo       (e.g., 192.168.1.105, 10.0.0.15)
echo.
echo  FIREWALL REQUIREMENTS:
echo    - Ensure Windows Defender Firewall allows incoming TCP on port 8000.
echo    - If prompted by Windows Defender Firewall, select "Allow Access"
echo      for Private Networks.
echo.
echo =========================================================================
echo.

python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
pause
