@echo off
setlocal

if not exist .venv (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 goto :error
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
python -m streamlit run app.py
goto :eof

:error
echo.
echo اجرای برنامه ناموفق بود. مطمئن شوید Python 3.11 نصب شده است.
pause
exit /b 1
