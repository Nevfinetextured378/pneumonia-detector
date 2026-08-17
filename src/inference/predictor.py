"""
推理预测模块
============
加载训练好的模型，对单张胸部X光图片进行预测。

使用方法:
    predictor = PneumoniaPredictor(model_path="models/best_model.pth")
    result = predictor.predict("path/to/chest_xray.jpg")
    # result = {"class": "PNEUMONIA", "probability": 0.923, ...}
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 修复 Windows 编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    DEVICE,
    IMAGE_SIZE,
    MEAN,
    STD,
    CLASS_NAMES,
    CLASS_LABELS_CN,
    PRED_THRESHOLD,
    MODEL_SAVE_PATH,
)
from src.models.classifier import PneumoniaClassifier


class PneumoniaPredictor:
    """
    肺炎检测推理器。

    Args:
        model_path: 训练好的模型权重路径
        device: 推理设备
    """

    def __init__(self, model_path=MODEL_SAVE_PATH, device=DEVICE):
        self.device = device

        # 加载模型
        self.model = PneumoniaClassifier()
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        # 预处理变换（与验证集一致，无数据增强）
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

        print(f"  模型已加载: {model_path}")
        if "epoch" in checkpoint:
            print(f"  训练轮次: {checkpoint['epoch']}")
        if "best_val_acc" in checkpoint:
            print(f"  最佳验证准确率: {checkpoint['best_val_acc']:.2%}")

    @torch.no_grad()
    def predict(self, image):
        """
        对图片进行预测。

        Args:
            image: PIL Image 对象 或 图片文件路径

        Returns:
            dict: {
                "class": "NORMAL" | "PNEUMONIA",
                "class_cn": "正常" | "肺炎",
                "probability": 0.923,  # 预测类别的概率
                "probabilities": {"NORMAL": 0.077, "PNEUMONIA": 0.923},
                "raw_logits": [logit_normal, logit_pneumonia],
            }
        """
        start_time = time.time()

        # 加载图片
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image 必须是 PIL Image 对象或文件路径")

        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

        # 获取预测结果
        probs = probabilities.cpu().numpy()
        pred_idx = probs.argmax()
        pred_prob = float(probs[pred_idx])

        inference_time = (time.time() - start_time) * 1000  # 转为毫秒

        return {
            "class": CLASS_NAMES[pred_idx],
            "class_cn": CLASS_LABELS_CN[CLASS_NAMES[pred_idx]],
            "probability": pred_prob,
            "probabilities": {
                CLASS_NAMES[0]: float(probs[0]),
                CLASS_NAMES[1]: float(probs[1]),
            },
            "raw_logits": logits.squeeze(0).cpu().tolist(),
            "inference_time_ms": round(inference_time, 2),
        }

    def get_input_tensor(self, image):
        """返回预处理后的 tensor（供 Grad-CAM 使用）"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image 必须是 PIL Image 对象或文件路径")

        return self.transform(image).unsqueeze(0).to(self.device)


def load_predictor(model_path=MODEL_SAVE_PATH, device=DEVICE):
    """工厂函数：创建推理器实例"""
    return PneumoniaPredictor(model_path, device)
