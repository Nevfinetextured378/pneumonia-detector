"""
模型定义模块
============
基于 ResNet-50 的迁移学习分类器。

迁移学习策略：
- 加载 ImageNet 预训练权重
- 冻结卷积层（feature extractor）
- 替换最后的全连接层为二分类头
- 只训练 ~200 万参数（全模型 ~2300 万参数）

架构:
    ResNet-50 Backbone (frozen)
        → Global Average Pooling
        → Dropout(0.3)
        → Linear(2048 → 2)
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import MODEL_NAME, NUM_CLASSES, PRETRAINED, FREEZE_BACKBONE, DEVICE


class PneumoniaClassifier(nn.Module):
    """
    基于 ResNet-50 的肺炎检测分类器。

    Args:
        num_classes: 分类数量（默认 2: NORMAL, PNEUMONIA）
        pretrained: 是否使用 ImageNet 预训练权重
        freeze_backbone: 是否冻结主干网络
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
    ):
        super(PneumoniaClassifier, self).__init__()

        # 加载预训练 ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)

        # 冻结骨干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 获取 backbone 最后一层的输出特征数 (ResNet-50 = 2048)
        in_features = self.backbone.fc.in_features

        # 替换分类头
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

        self._initialize_classifier_head()

    def _initialize_classifier_head(self):
        """用 Xavier 初始化新的分类头"""
        for module in self.backbone.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.backbone(x)

    def get_feature_maps(self, x):
        """
        获取 layer4 的特征图（用于 Grad-CAM）。

        Returns:
            feature_maps: layer4 输出的特征图 [B, 2048, 7, 7]
        """
        # ResNet-50 的各层:
        # conv1 → bn1 → relu → maxpool
        # → layer1 → layer2 → layer3 → layer4
        # → avgpool → fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # <-- 这是我们要的特征图

        return x

    def forward_from_features(self, features):
        """从特征图计算最终分类结果"""
        x = self.backbone.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x

    def get_layer4(self):
        """返回 layer4 模块（Grad-CAM 注册 hook 用）"""
        return self.backbone.layer4


def create_model(device=DEVICE):
    """
    工厂函数：创建模型并移动到指定设备。

    Args:
        device: 目标设备

    Returns:
        PneumoniaClassifier 实例
    """
    model = PneumoniaClassifier()
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  模型: {MODEL_NAME}")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    print(f"  设备: {device}")

    return model
