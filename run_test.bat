@echo off
chcp 65001 >nul
REM ============================================================
REM 测试已训练好的模型（PolitiFact 验证集）
REM ============================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv，请先运行 setup.bat
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

REM 默认走 train_config.py 跑出来的输出目录
set TEST_CSV=outputs\politifact_config\splits\politifact_val.csv
set CKPT=outputs\politifact_config\checkpoints\best.pt

if not exist "%TEST_CSV%" (
    echo [警告] 找不到 %TEST_CSV%
    echo 请先训练，或自行修改本 .bat 里的 TEST_CSV / CKPT 路径
    pause
    exit /b 1
)

if not exist "%CKPT%" (
    echo [警告] 找不到 checkpoint %CKPT%
    pause
    exit /b 1
)

echo ============================================================
echo 测试集：%TEST_CSV%
echo 模型 ：%CKPT%
echo ============================================================
python test.py ^
    --test_csv "%TEST_CSV%" ^
    --checkpoint "%CKPT%" ^
    --device cpu

echo.
pause
