# ============================================================
# IBSR-18 脑分割项目 —— 一键环境配置脚本（Windows / PowerShell）
# 用法：在项目根目录运行
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# 或直接双击项目根目录的「一键配置环境.bat」
# ============================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step($msg) { Write-Host "`n========== $msg ==========" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "[!]  $msg" -ForegroundColor Yellow }

# ------------------------------------------------------------
# 1. 检查 Python 3.11
# ------------------------------------------------------------
Write-Step "1/6 检查 Python"

$pyCmd = $null
foreach ($candidate in @("python", "py -3.11")) {
    $ver = & cmd /c "$candidate --version 2>nul"
    if ($ver -match "Python 3\.11\.(\d+)") {
        $pyCmd = $candidate
        Write-Ok "找到 $ver"
        break
    }
}

if (-not $pyCmd) {
    Write-Warn2 "未找到 Python 3.11.x。"
    Write-Host "    请先安装 Python 3.11（安装时务必勾选 Add Python to PATH）："
    Write-Host "    https://www.python.org/downloads/release/python-3119/"
    Write-Host "    安装后重新运行本脚本。"
    Read-Host "按回车键退出"
    exit 1
}

# ------------------------------------------------------------
# 2. 创建虚拟环境
# ------------------------------------------------------------
Write-Step "2/6 创建虚拟环境 .venv"

if (Test-Path ".venv\Scripts\python.exe") {
    Write-Ok ".venv 已存在，跳过创建"
} else {
    & cmd /c "$pyCmd -m venv .venv"
    if ($LASTEXITCODE -ne 0) { Write-Warn2 "venv 创建失败"; exit 1 }
    Write-Ok "虚拟环境创建完成"
}

$VenvPy = ".\.venv\Scripts\python.exe"

# ------------------------------------------------------------
# 3. 升级 pip
# ------------------------------------------------------------
Write-Step "3/6 升级 pip"
& $VenvPy -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# 4. 安装 PyTorch（关键：GPU 版必须走官方 index，不能用国内镜像）
# ------------------------------------------------------------
Write-Step "4/6 安装 PyTorch"

$hasNvidia = $false
$nvsmi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
if ($nvsmi) {
    $hasNvidia = $true
    Write-Ok "检测到 NVIDIA GPU：$nvsmi"
}

$torchInstalled = (& $VenvPy -c "import torch; print(torch.__version__)" 2>$null)
if ($torchInstalled) {
    Write-Ok "PyTorch 已安装：$torchInstalled，跳过（若为 CPU 版且你有 N 卡，请删除 .venv 后重跑）"
} elseif ($hasNvidia) {
    Write-Host "    安装 CUDA 12.6 版 torch（约 2.5GB，耗时取决于网速）..."
    & $VenvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Warn2 "未检测到 NVIDIA GPU，安装 CPU 版 torch（训练慢，但网页推理可用）"
    & $VenvPy -m pip install torch torchvision
}

# ------------------------------------------------------------
# 5. 安装项目全部依赖（含网页 web 和开发 dev）
# ------------------------------------------------------------
Write-Step "5/6 安装项目依赖（ibsr_unet + web + dev）"
& $VenvPy -m pip install -e ".[web,dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Warn2 "依赖安装失败，可重试。注意：切勿对 torch 加国内镜像（会变成 CPU 版）。"
    Read-Host "按回车键退出"
    exit 1
}

# ------------------------------------------------------------
# 6. 验证安装
# ------------------------------------------------------------
Write-Step "6/6 验证"

$verifyCode = "import torch,fastapi,uvicorn,SimpleITK,monai,nibabel;print('PyTorch:',torch.__version__);print('CUDA:',torch.cuda.is_available());print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only');print('FastAPI:',fastapi.__version__);print('MONAI:',monai.__version__);print('SimpleITK:',SimpleITK.__version__)"
& $VenvPy -c $verifyCode

if ($LASTEXITCODE -ne 0) {
    Write-Warn2 "验证失败：上面有模块导入错误，请截图发到群里。"
    Read-Host "按回车键退出"
    exit 1
}

Write-Host ""
Write-Ok "环境配置完成！"
Write-Host ""
Write-Host "接下来：" -ForegroundColor White
Write-Host "  1) 把 ibsr18-team-resources.zip 在【项目根目录】直接解压（自动落到 data\ 和 outputs\）"
Write-Host "  2) 双击「启动网页.bat」，或运行 .\.venv\Scripts\python.exe web\server.py"
Write-Host "  3) 浏览器打开 http://127.0.0.1:8000"
Write-Host ""
Read-Host "按回车键退出"
