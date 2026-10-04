@echo off
REM ============================================================
REM  FI-Lab + Mechanical Ventilation COPD
REM  In-hospital mortality risk prediction  -  local launcher
REM
REM  Double-click this file to start the app, then open
REM  http://127.0.0.1:8501 in a browser.
REM
REM  Note: the proxy variables are cleared on purpose. A stray
REM  HTTP_PROXY in the environment prevents Streamlit from binding
REM  to localhost on some Windows setups.
REM ============================================================

set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set ALL_PROXY=

cd /d "%~dp0"

echo Starting the prediction app...
echo When you see "You can now view your Streamlit app", open:
echo    http://127.0.0.1:8501
echo.
echo Press Ctrl+C in this window to stop the app.
echo.

python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false

pause
