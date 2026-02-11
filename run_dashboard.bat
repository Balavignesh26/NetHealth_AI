@echo off
echo Starting Belden ONE View Dashboard...
call venv\Scripts\activate.bat
streamlit run src\dashboard\app.py
pause
