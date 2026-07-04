@echo off
chcp 65001 >nul
REM ============================================================
REM 训练 GossipCop 数据集（使用 YAML 配置）
REM ============================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv，请先运行 setup.bat
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo ============================================================
echo 开始训练 GossipCop（配置文件：configs/gossipcop_smoke.yaml）
echo ============================================================
python train_config.py --config configs/gossipcop_smoke.yaml

echo.
echo ============================================================
echo 训练结束
echo ============================================================
pause
