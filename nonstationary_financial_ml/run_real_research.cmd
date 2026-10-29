@echo off
cd /d "%~dp0"
uv sync --dev
if errorlevel 1 exit /b %errorlevel%
uv run pytest -q
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\prepare_data.py
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\build_features.py
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\train_models.py
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\evaluate_models.py
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\analyze_drift.py
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\generate_report.py
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\export_signal.py
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\verify_notebooks.py
if errorlevel 1 exit /b %errorlevel%
uv run python scripts\verify_artifacts.py --require-notebooks
if errorlevel 1 exit /b %errorlevel%
echo Completed real historical research. See outputs\research_report.md.
