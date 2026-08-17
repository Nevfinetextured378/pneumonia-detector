@echo off
chcp 65001 >nul
title ChestLens — 肺炎检测系统

echo ============================================
echo   ChestLens 一键启动
echo   胸部X光肺炎AI辅助检测系统
echo ============================================
echo.

:: 检查 Python 环境
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 检查模型文件
if not exist "models\best_model.pth" (
    echo [警告] 模型文件不存在！
    echo 请先运行: python scripts/train.py
    echo.
    choice /c yn /m "是否仍然启动后端？"
    if errorlevel 2 exit /b 1
)

echo.
echo [1/2] 启动 FastAPI 后端 (端口 8000)...
start "ChestLens-API" cmd /k "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

:: 等待后端启动
echo        等待后端启动中...
timeout /t 5 /nobreak >nul

echo [2/2] 启动 React 前端 (端口 3000)...
cd frontend

:: 检查 node_modules
if not exist "node_modules\" (
    echo        首次运行，安装前端依赖...
    call npm install
)

start "ChestLens-Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ============================================
echo   启动完成！
echo   前端: http://localhost:3000
echo   后端: http://localhost:8000
echo   API文档: http://localhost:8000/docs
echo ============================================
echo.
echo 按任意键关闭此窗口（不会关闭服务）
pause >nul
