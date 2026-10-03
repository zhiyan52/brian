@echo off
chcp 65001 >nul
REM ============================================================
REM  IBSR-18 项目一键环境配置（Windows 双击运行）
REM ============================================================
cd /d "%~dp0"
echo 正在配置 IBSR-18 项目环境，请保持联网...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_env.ps1"
