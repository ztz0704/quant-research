@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\run_research_v2.py --config configs\research_v2_historical.yaml
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe scripts\run_research_v2.py --config configs\research_v2_historical.yaml --verify
exit /b %errorlevel%
