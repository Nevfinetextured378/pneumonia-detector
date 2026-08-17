"""
Grad-CAM 可视化模块
===================
实现 Gradient-weighted Class Activation Mapping (Grad-CAM)，
用于可视化模型在做出预测时关注的图像区域。

原理:
1. 前向传播：获取目标层（layer4）的特征图
2. 反向传播：计算目标类对特征图的梯度
3. 全局平均池化梯度 → 得到每个通道的重要性权重
4. 加权求和特征图 → 得到类激活图 (CAM)
5. ReLU → 上采样到原始尺寸 → 伪彩色映射
6. 热力图叠加到原图上

参考论文:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017
"""

import io
import base64
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

# 修复 Windows 编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import GRADCAM_TARGET_LAYER, DEVICE


class GradCAM:
    """
    Grad-CAM 生成器。

    使用方法:
        gradcam = GradCAM(model)
        cam = gradcam.generate(input_tensor, target_class=1)  # 肺炎类
        overlay = gradcam.overlay(original_image, cam)        # 叠加到原图

    使用 torch.autograd.grad 方法，比 hook 方式更可靠。
    """

    def __init__(self, model):
        """
        Args:
            model: PneumoniaClassifier 模型实例
        """
        self.model = model

    def generate(self, input_tensor, target_class=None):
        """
        生成 Grad-CAM 热力图。

        使用 torch.autograd.grad 直接计算目标分数对特征图的梯度，
        不需要注册 hook。

        Args:
            input_tensor: 预处理后的输入张量 [1, 3, H, W]
            target_class: 目标类别索引。None 则使用模型预测的类别

        Returns:
            cam: numpy array [H, W]，值范围 [0, 1]
        """
        # ---- 前向传播 ----
        self.model.eval()

        # 临时启用骨干网络的梯度计算（Grad-CAM 需要梯度）
        # 保存原始 requires_grad 状态并在结束时恢复
        original_states = {}
        for name, param in self.model.backbone.named_parameters():
            original_states[name] = param.requires_grad
            param.requires_grad_(True)

        # 获取 layer4 的特征图（保留在计算图中）
        feature_maps = self.model.get_feature_maps(input_tensor)  # [1, 2048, 7, 7]

        # 从特征图计算最终分类
        output = self.model.forward_from_features(feature_maps)  # [1, 2]

        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # ---- 计算梯度 ----
        self.model.zero_grad()

        # 目标类的 score
        score = output[0, target_class]

        # 计算 score 对 feature_maps 的梯度
        grads = torch.autograd.grad(
            score, feature_maps, retain_graph=False, create_graph=False
        )[0]  # [1, 2048, 7, 7]

        # 恢复原始 requires_grad 状态
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad_(original_states[name])

        # ---- 计算 Grad-CAM ----
        # 1. 对梯度做全局平均池化 → 得到每个通道的权重
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 2. 加权求和特征图
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)  # [1, 1, 7, 7]

        # 3. ReLU：只保留对目标类有正向贡献的区域
        cam = torch.relu(cam)

        # 4. 上采样到输入尺寸
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # 5. 归一化到 [0, 1]
        cam = cam.squeeze().cpu().detach().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay(self, original_image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        将 Grad-CAM 热力图叠加到原始图片上。

        Args:
            original_image: PIL Image 或 numpy array (H, W, 3) RGB
            cam: Grad-CAM 热力图 [H, W]，值范围 [0, 1]
            alpha: 热力图透明度 (0=完全透明, 1=完全不透明)
            colormap: OpenCV 颜色映射

        Returns:
            numpy array (H, W, 3) RGB 叠加后的图片
        """
        # 确保原图是 numpy array
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        # 确保是 RGB 格式
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)

        h, w = original_image.shape[:2]

        # 将 CAM 上采样到原始图片尺寸
        cam_resized = cv2.resize(cam, (w, h))

        # 应用伪彩色映射
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加：原图 * (1-alpha) + 热力图 * alpha
        overlay = np.uint8(
            original_image * (1 - alpha) + heatmap * alpha
        )

        return overlay

    def overlay_to_base64(
        self, original_image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET
    ):
        """
        生成叠加图并编码为 Base64 字符串（用于 API 返回）。

        Returns:
            str: "data:image/png;base64,iVBORw0KGgo..."
        """
        overlay = self.overlay(original_image, cam, alpha, colormap)
        overlay_pil = Image.fromarray(overlay)

        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        buffer.seek(0)

        b64_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    def image_to_base64(self, image):
        """将 PIL Image 或 numpy array 编码为 Base64"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        b64_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    def __del__(self):
        """析构函数"""
        pass
