"""
全局配置
========
集中管理项目中的所有路径、超参数和常量。

使用方法:
    from src.config import DATA_DIR, BATCH_SIZE, ...
"""

import os
import torch

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chest_xray")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# 数据集子目录
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# ============================================================
# 设备配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 模型参数
# ============================================================
MODEL_NAME = "resnet50"                     # 预训练模型名称
NUM_CLASSES = 2                             # 二分类: NORMAL / PNEUMONIA
PRETRAINED = True                           # 使用 ImageNet 预训练权重
FREEZE_BACKBONE = True                      # 是否冻结骨干网络（迁移学习第一阶段）

# ============================================================
# 训练超参数
# ============================================================
BATCH_SIZE = 32                             # 批大小（可根据内存调整）
NUM_EPOCHS = 15                             # 训练轮数（迁移学习通常 10-20 轮）
LEARNING_RATE = 1e-4                        # 学习率（迁移学习用较小值）
WEIGHT_DECAY = 1e-4                         # 权重衰减（L2 正则化）
NUM_WORKERS = 0                             # 数据加载线程数（Windows 下建议 0）

# 学习率调度
LR_STEP_SIZE = 5                            # 每隔多少轮降低学习率
LR_GAMMA = 0.5                              # 学习率衰减因子

# ============================================================
# 数据预处理参数
# ============================================================
IMAGE_SIZE = 224                            # 输入图片大小（ResNet 标准输入）
MEAN = [0.485, 0.456, 0.406]               # ImageNet 均值（RGB）
STD = [0.229, 0.224, 0.225]                # ImageNet 标准差（RGB）

# ============================================================
# 推理参数
# ============================================================
PRED_THRESHOLD = 0.5                        # 分类阈值（概率 > 0.5 判为肺炎）
GRADCAM_TARGET_LAYER = "layer4"             # ResNet50 最后一个 bottleneck block

# ============================================================
# 模型保存
# ============================================================
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.pth")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# ============================================================
# 类别映射
# ============================================================
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_LABELS_CN = {"NORMAL": "正常", "PNEUMONIA": "肺炎"}
