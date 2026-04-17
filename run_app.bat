@echo off
REM Run Hallucination Detector Streamlit App
echo Starting Hallucination Detector UI...
echo.
cd /d "%~dp0"
C:/Users/HP/AppData/Local/Programs/Python/Python312/python.exe -m streamlit run app.py
pause
