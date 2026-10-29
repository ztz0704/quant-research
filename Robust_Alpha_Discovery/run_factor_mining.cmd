@echo off
cd /d "%~dp0"
uv run python scripts\mine_factors.py mine --config configs\mine_real.yaml
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\mine_factors.py evaluate --config configs\mine_real.yaml
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\verify_mining.py --config configs\mine_real.yaml
if errorlevel 1 exit /b %errorlevel%
echo Completed real-data factor mining. Read reports\generated\factor_mining\mining_report.md
