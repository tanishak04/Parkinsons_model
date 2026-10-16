@echo off
title Parkinson's AI Diagnostic Dashboard
cd /d "%~dp0"
echo =======================================================
echo Launching Parkinson's Disease Diagnostic Dashboard...
echo Opening browser: http://localhost:8501
echo =======================================================
start "" http://localhost:8501
"C:\Users\mohan\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
pause
