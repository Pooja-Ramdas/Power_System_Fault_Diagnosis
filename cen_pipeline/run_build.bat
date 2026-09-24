@echo off
REM Launch build_dataset.py fully detached from any parent process tree.
REM Output goes to build_log.txt (unbuffered via python -u).
REM After build completes, regenerate_manifest.py runs to create a complete manifest.

set VENV_PYTHON=D:\Downloads\Power_System_Fault_Diagnosis\.venv\Scripts\python.exe
set PIPELINE_DIR=D:\Downloads\Power_System_Fault_Diagnosis\cen_pipeline
set LOG_FILE=%PIPELINE_DIR%\build_log.txt

echo ============================================== >> "%LOG_FILE%"
echo Build started at %date% %time% >> "%LOG_FILE%"
echo ============================================== >> "%LOG_FILE%"

"%VENV_PYTHON%" -u "%PIPELINE_DIR%\build_dataset.py" >> "%LOG_FILE%" 2>&1

echo ============================================== >> "%LOG_FILE%"
echo Build finished at %date% %time% (exit code %ERRORLEVEL%) >> "%LOG_FILE%"
echo ============================================== >> "%LOG_FILE%"

"%VENV_PYTHON%" -u "%PIPELINE_DIR%\regenerate_manifest.py" >> "%LOG_FILE%" 2>&1

echo [run_build] ALL DONE at %date% %time% >> "%LOG_FILE%"
