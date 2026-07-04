@echo off
chcp 65001 >nul
REM ============================================================
REM 小样本 smoke test（直接用 train.py 命令行参数）
REM 适合验证训练流程能否跑通
REM ============================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv，请先运行 setup.bat
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo ============================================================
echo PolitiFact smoke test（20 条样本，1 个 epoch，CPU）
echo ============================================================
python train.py ^
    --csv data/processed/politifact.csv ^
    --dataset_name politifact ^
    --max_samples 20 ^
    --epochs 1 ^
    --batch_size 1 ^
    --device cpu

echo.
pause
