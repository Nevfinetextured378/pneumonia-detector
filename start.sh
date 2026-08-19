#!/bin/bash
# ============================================
#   ChestLens 一键启动脚本 (Linux/Mac)
#   胸部X光肺炎AI辅助检测系统
# ============================================

set -e

echo "============================================"
echo "  ChestLens 一键启动"
echo "  胸部X光肺炎AI辅助检测系统"
echo "============================================"
echo ""

# 检查模型文件
if [ ! -f "models/best_model.pth" ]; then
    echo "[警告] 模型文件不存在！"
    echo "请先运行: python scripts/train.py"
    read -p "是否仍然启动后端？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[环境] 已激活虚拟环境"
fi

echo ""
echo "[1/2] 启动 FastAPI 后端 (端口 8000)..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "      后端 PID: $BACKEND_PID"

sleep 3

echo "[2/2] 启动 React 前端 (端口 5173)..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "      首次运行，安装前端依赖..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================"
echo "  启动完成！"
echo "  前端: http://localhost:5173"
echo "  后端: http://localhost:8000"
echo "  API文档: http://localhost:8000/docs"
echo "============================================"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 捕获退出信号，清理子进程
cleanup() {
    echo ""
    echo "正在关闭服务..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "已关闭所有服务"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 等待子进程
wait
