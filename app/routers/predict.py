"""
预测 API 路由
=============
POST /api/predict — 上传胸部X光图片，返回预测结果和 Grad-CAM 热力图

请求: multipart/form-data (file)
响应: JSON (预测类别、概率、热力图 Base64)
"""

import io
import sys
import time
import traceback

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# 修复 Windows 编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.inference.predictor import PneumoniaPredictor
from src.inference.gradcam import GradCAM
from src.config import MODEL_SAVE_PATH, DEVICE, CLASS_NAMES, CLASS_LABELS_CN

router = APIRouter(prefix="/api", tags=["prediction"])

# 全局模型实例（在应用启动时初始化）
predictor: PneumoniaPredictor = None

# 允许的图片格式
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/bmp",
    "image/tiff",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def get_predictor() -> PneumoniaPredictor:
    """获取全局预测器实例"""
    global predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="模型尚未加载，请稍后重试",
        )
    return predictor


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    上传胸部X光图片进行肺炎检测。

    请求:
        - file: 图片文件 (JPEG/PNG, 最大 10MB)

    返回:
        JSON 包含预测类别、概率、Grad-CAM 热力图
    """
    # 1. 文件校验
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式: {file.content_type}。请上传 JPEG/PNG 格式的图片。",
        )

    # 读取文件内容
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="无法读取上传的文件")

    # 检查文件大小
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大 ({len(contents) / 1024 / 1024:.1f} MB)，最大支持 10 MB",
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="上传的文件为空")

    # 2. 图片预处理
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="无法解析图片文件，请确认上传的是有效的胸部X光图片",
        )

    # 3. 模型推理
    try:
        model = get_predictor()

        # 预测
        start_time = time.time()
        result = model.predict(image)
        inference_time = (time.time() - start_time) * 1000

        # 生成 Grad-CAM
        gradcam = GradCAM(model.model)
        input_tensor = model.get_input_tensor(image)

        # 对预测类别生成热力图
        target_class = 1 if result["class"] == "PNEUMONIA" else 0
        cam = gradcam.generate(input_tensor, target_class=target_class)

        # 编码为 Base64
        gradcam_b64 = gradcam.overlay_to_base64(image, cam, alpha=0.4)
        original_b64 = gradcam.image_to_base64(image)

        # 4. 构建响应
        response_data = {
            "success": True,
            "prediction": {
                "class": result["class"],
                "class_cn": result["class_cn"],
                "probability": result["probability"],
                "confidence": result["probability"] * 100,
            },
            "probabilities": result["probabilities"],
            "gradcam_image": gradcam_b64,
            "original_image": original_b64,
            "inference_time_ms": round(inference_time, 2),
        }

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"模型文件未找到: {str(e)}。请先运行训练脚本生成模型。",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"推理过程出错: {str(e)}",
        )


@router.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        model = get_predictor()
        return {
            "status": "ok",
            "model_loaded": True,
            "device": str(DEVICE),
        }
    except HTTPException:
        return {
            "status": "starting",
            "model_loaded": False,
        }
