@echo off
rem Start the app on Windows:  scripts\run.bat   ->  http://localhost:8000
cd /d "%~dp0\.."
set PYTHONPATH=backend
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
