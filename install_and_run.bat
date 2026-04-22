@echo off
title Baby Soother v3
color 0A
echo.
echo  ==============================================
echo    Baby Soother v3  ^|  ML + Voice Cloning
echo  ==============================================
echo.

python --version >nul 2>&1 || (echo  ERROR: Python not found. && pause && exit /b 1)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  Python %%v detected

if not exist "venv\" (
    echo  Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo.
echo  Installing packages (first time ~2-3 minutes)...
pip install --quiet --upgrade pip
pip install --quiet flask flask-cors gTTS numpy scipy librosa scikit-learn soundfile sounddevice requests standard-aifc standard-sunau

echo.
echo  ✓ Ready!
echo.
echo  ==============================================
echo    Open in browser:  http://localhost:5050
echo  ==============================================
echo.
python app.py
pause
