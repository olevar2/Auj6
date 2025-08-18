@echo off
echo ==========================================
echo AUJ Platform Economic Calendar Migration
echo ==========================================
echo.

cd /d "%~dp0.."

echo Running database migration for economic calendar...
py scripts\migrate_economic_calendar_db.py

echo.
echo Migration script completed.
echo Check the logs above for results.
echo.
pause