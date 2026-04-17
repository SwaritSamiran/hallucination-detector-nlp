# Run Hallucination Detector Streamlit App
Write-Host "Starting Hallucination Detector UI..." -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

$python = "C:/Users/HP/AppData/Local/Programs/Python/Python312/python.exe"

Write-Host "Using Python: $python" -ForegroundColor Gray
Write-Host "Launching Streamlit app..." -ForegroundColor Cyan
Write-Host ""

& $python -m streamlit run app.py
