@echo off
rem ===========================================================================
rem thenoise.bat - Thin wrapper around thenoise.ps1 so the bootstrap is
rem double-click friendly and works from cmd. All arguments are forwarded.
rem ===========================================================================
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%thenoise.ps1" %*
exit /b %errorlevel%
