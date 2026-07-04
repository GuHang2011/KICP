@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ============================================================
REM KICP 一键环境安装脚本（Windows）
REM 自动完成：
REM   1. 检测 Python（缺失则用 winget 自动安装 Python 3.11）
REM   2. 清理已损坏的 .venv
REM   3. 创建新的 .venv 虚拟环境
REM   4. 检测 NVIDIA GPU，自动选择 CPU / CUDA 版 torch
REM   5. 安装 requirements.txt 其余依赖
REM   6. 配置 HuggingFace 镜像（加速国内下载）
REM   7. 可选：预下载 CLIP 模型（~1.7GB）
REM   8. 验证安装
REM ============================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo  KICP 一键环境安装脚本
echo  项目目录: %CD%
echo ============================================================
echo.

REM ------------------------------------------------------------
REM [1/8] 检测 Python
REM ------------------------------------------------------------
echo [1/8] 检测 Python ...
set "PY_CMD="

for %%P in (py python python3) do (
    if not defined PY_CMD (
        %%P --version >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD=%%P"
        )
    )
)

if not defined PY_CMD (
    echo.
    echo [警告] 未检测到 Python，尝试用 winget 自动安装 Python 3.11 ...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo [错误] 系统没有 winget，也没有 Python
        echo 请手动安装 Python 3.10 或 3.11: https://www.python.org/downloads/
        echo 安装时务必勾选 "Add Python to PATH"
        pause
        exit /b 1
    )
    winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo [错误] winget 安装 Python 失败，请手动安装
        pause
        exit /b 1
    )
    echo.
    echo [提示] Python 安装完成，请关闭本窗口，重新打开后再运行 setup.bat
    pause
    exit /b 0
)

for /f "tokens=2 delims= " %%V in ('%PY_CMD% --version 2^>^&1') do set "PY_VER=%%V"
echo     检测到 Python %PY_VER% (命令: %PY_CMD%)

REM 简单校验主版本 3.x
echo %PY_VER% | findstr /r "^3\.[91][012]*\." >nul
if errorlevel 1 (
    echo [警告] 当前 Python 版本 %PY_VER%，建议使用 3.10 / 3.11
    echo        3.9 / 3.12 也能跑，但 3.12 部分包可能装不上
    echo.
    choice /c YN /n /m "是否继续? [Y/N]: "
    if errorlevel 2 exit /b 1
)

REM ------------------------------------------------------------
REM [2/8] 清理已损坏的 .venv
REM ------------------------------------------------------------
echo.
echo [2/8] 检查 .venv ...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe --version >nul 2>&1
    if errorlevel 1 (
        echo     检测到 .venv 已损坏，正在删除 ...
        rmdir /s /q ".venv"
    ) else (
        echo     .venv 健康，跳过重建
        goto :venv_ready
    )
) else if exist ".venv" (
    echo     .venv 目录存在但不完整，删除重建 ...
    rmdir /s /q ".venv"
)

REM ------------------------------------------------------------
REM [3/8] 创建虚拟环境
REM ------------------------------------------------------------
echo.
echo [3/8] 创建虚拟环境 .venv ...
%PY_CMD% -m venv .venv
if errorlevel 1 (
    echo [错误] 创建虚拟环境失败
    pause
    exit /b 1
)

:venv_ready

REM 激活
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [错误] 激活 .venv 失败
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM [4/8] 升级 pip 并配置国内镜像
REM ------------------------------------------------------------
echo.
echo [4/8] 升级 pip 并配置清华镜像 ...
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1

REM ------------------------------------------------------------
REM [5/8] 检测 GPU 并安装对应版本的 PyTorch
REM ------------------------------------------------------------
echo.
echo [5/8] 检测 NVIDIA GPU ...
set "TORCH_INDEX="
set "GPU_INFO="

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv,noheader 2^>nul') do (
        if not defined GPU_INFO set "GPU_INFO=%%i"
    )
)

if defined GPU_INFO (
    echo     检测到 GPU: %GPU_INFO%
    echo.
    echo     请选择 CUDA 版本:
    echo       [1] CUDA 12.1   (推荐，新驱动)
    echo       [2] CUDA 11.8   (老驱动)
    echo       [3] CPU 版本    (不用 GPU)
    choice /c 123 /n /m "请选择 [1/2/3]: "
    if errorlevel 3 (
        set "TORCH_INDEX="
    ) else if errorlevel 2 (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
    ) else (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    )
) else (
    echo     未检测到 NVIDIA GPU，将安装 CPU 版本 PyTorch
)

echo.
echo     安装 PyTorch ...
if defined TORCH_INDEX (
    pip install torch torchvision --index-url !TORCH_INDEX!
) else (
    pip install "torch>=2.1.0" "torchvision>=0.16.0"
)

if errorlevel 1 (
    echo [错误] PyTorch 安装失败
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM [6/8] 安装其他依赖
REM ------------------------------------------------------------
echo.
echo [6/8] 安装其他依赖 (requirements.txt) ...
pip install transformers>=4.38.0 pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 tqdm>=4.66.0 pyyaml>=6.0.0 pillow>=10.0.0 requests>=2.31.0
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM [7/8] 可选：预下载 CLIP 模型
REM ------------------------------------------------------------
echo.
echo [7/8] 是否预下载 CLIP 模型 (openai/clip-vit-large-patch14-336，约 1.7GB) ?
echo     选 Y 现在下载（占用空间，但首次训练不卡）
echo     选 N 跳过（首次训练时自动下载）
choice /c YN /n /m "请选择 [Y/N]: "
if errorlevel 2 goto :skip_download

echo.
echo     配置 HuggingFace 国内镜像 hf-mirror.com ...
set "HF_ENDPOINT=https://hf-mirror.com"
echo     开始下载 (国内镜像加速) ...
python -c "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; from transformers import CLIPModel, CLIPProcessor; m='openai/clip-vit-large-patch14-336'; CLIPProcessor.from_pretrained(m); CLIPModel.from_pretrained(m); print('CLIP 模型已下载到本地缓存')"
if errorlevel 1 (
    echo [警告] CLIP 模型下载失败，首次训练时会重新尝试
)

:skip_download

REM ------------------------------------------------------------
REM [8/8] 验证安装
REM ------------------------------------------------------------
echo.
echo [8/8] 验证安装 ...
python -c "import torch, transformers, pandas, sklearn, yaml, PIL; print('torch       :', torch.__version__); print('CUDA 可用   :', torch.cuda.is_available()); print('transformers:', transformers.__version__); print('pandas      :', pandas.__version__); print('sklearn     :', sklearn.__version__)"

if errorlevel 1 (
    echo [错误] 验证失败，请检查上面的报错信息
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  安装完成！
echo ============================================================
echo  接下来你可以双击运行：
echo    run_train_smoke.bat       - 20 条样本 smoke test
echo    run_train_politifact.bat  - 训练 PolitiFact
echo    run_train_gossipcop.bat   - 训练 GossipCop
echo    run_test.bat              - 测试已训练模型
echo ============================================================
echo.
pause
endlocal
