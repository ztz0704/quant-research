@echo off
setlocal
cd /d "%~dp0\.."
.venv\Scripts\python.exe scripts\run_research_v3.py --config configs/research_v3_engineering.yaml
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe scripts\run_research_v3.py --config configs/research_v3_historical.yaml
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe scripts\run_research_v3.py --config configs/research_v3_engineering.yaml --verify
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe scripts\run_research_v3.py --config configs/research_v3_historical.yaml --verify
exit /b %errorlevel%
