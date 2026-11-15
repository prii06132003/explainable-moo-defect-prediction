@echo off
title Explainable MOO Software Defect Prediction
echo ============================================================
echo  Starting Software Defect Prediction Application...
echo ============================================================
cd /d "%~dp0"

:: Check if already running on port 8501
netstat -ano | findstr :8501 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo [INFO] Application is already running on http://localhost:8501!
    start http://localhost:8501
    exit /b
)

echo [INFO] Launching Streamlit server on http://localhost:8501...
start "" http://localhost:8501
python -m streamlit run app.py --server.port 8501
pause
