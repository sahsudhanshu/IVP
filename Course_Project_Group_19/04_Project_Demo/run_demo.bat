@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\..\.model_test_deps" if exist "%USERPROFILE%\miniconda3\envs\matsya\python.exe" (
  set "PYTHONPATH=%~dp0..\..\.model_test_deps"
  "%USERPROFILE%\miniconda3\envs\matsya\python.exe" "%~dp0demo.py"
  if errorlevel 1 pause
  exit /b
)
python "%~dp0demo.py"
if errorlevel 1 (
  echo Install Python and run: python -m pip install -r requirements.txt
  pause
)
