@echo off
echo ====================================================
echo  Installing Used Car Price Prediction Dependencies
echo ====================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not added to PATH.
    pause
    exit /b %errorlevel%
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment (venv)...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies from requirements.txt
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ====================================================
echo  All dependencies installed successfully!
echo  To start the server, run:
echo    venv\Scripts\activate
echo    uvicorn app.main:app --reload
echo ====================================================
pause
