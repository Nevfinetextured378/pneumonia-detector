"""
FastAPI 主应用
==============
胸部X光肺炎检测系统的后端服务。

启动方式:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

API 文档:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 修复 Windows 编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.predictor import PneumoniaPredictor
from src.config import MODEL_SAVE_PATH, DEVICE
from app.routers import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：
    - 启动时：加载模型到内存（全局共享）
    - 关闭时：清理资源
    """
    print("\n" + "=" * 60)
    print("  肺炎检测系统 — FastAPI 后端启动")
    print("=" * 60)

    # 启动时加载模型
    try:
        print(f"\n  正在加载模型: {MODEL_SAVE_PATH}")
        print(f"  推理设备: {DEVICE}")

        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"\n  ❌ 模型文件不存在: {MODEL_SAVE_PATH}")
            print(f"  请先运行: python scripts/train.py")
            raise FileNotFoundError(f"模型文件不存在: {MODEL_SAVE_PATH}")

        predict.predictor = PneumoniaPredictor(
            model_path=MODEL_SAVE_PATH, device=DEVICE
        )
        print(f"\n  ✅ 模型加载成功！")
        print(f"  API 文档: http://localhost:8000/docs")

    except Exception as e:
        print(f"\n  ❌ 模型加载失败: {e}")
        print(f"  请先运行: python scripts/train.py")
        # 不阻止应用启动，允许访问健康检查端点

    print("=" * 60 + "\n")

    yield  # 应用运行中...

    # 关闭时清理
    print("\n  正在关闭服务...")
    predict.predictor = None


# 创建 FastAPI 应用
app = FastAPI(
    title="胸部X光肺炎AI辅助检测系统",
    description="""
## 基于深度学习的医学影像辅助诊断

**功能:**
- 上传胸部X光图片
- 自动检测肺炎（正常/肺炎二分类）
- Grad-CAM 热力图可视化（显示AI关注的影像区域）

**技术栈:** PyTorch + ResNet-50 + FastAPI + React

**⚠️ 免责声明:** 本工具仅供科研/教育用途，不能替代专业医生诊断。
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",     # Vite 开发服务器
        "http://127.0.0.1:5173",
        "http://localhost:3000",     # React 开发服务器（备用）
        "http://127.0.0.1:3000",
        "http://localhost:3002",     # 端口冲突时 Vite 自动切换
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(predict.router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "胸部X光肺炎AI辅助检测系统",
        "version": "0.1.0",
        "docs": "/docs",
        "api": "/api/predict",
        "health": "/api/health",
    }
