@echo off
chcp 65001 >nul
REM ============================================================
REM  IBSR-18 脑萎缩定量分析平台 —— 启动网页
REM  浏览器访问 http://127.0.0.1:8000
REM ============================================================
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv，请先双击运行「一键配置环境.bat」
    pause
    exit /b 1
)
echo 正在启动服务，首次加载模型约需 30 秒，请稍候...
echo 浏览器访问: http://127.0.0.1:8000 （关闭此窗口即停止服务）
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" web\server.py
pause
