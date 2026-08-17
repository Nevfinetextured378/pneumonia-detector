#!/usr/bin/env python
"""
训练入口脚本
============
命令行运行: python scripts/train.py

运行前的准备:
1. 确保数据集已下载到 data/chest_xray/
2. 确保 PyTorch CUDA 版本已安装
3. 检查 src/config.py 中的超参数

训练流程:
1. 创建 DataLoader（含数据增强和类别权重）
2. 初始化 ResNet-50 迁移学习模型
3. 训练 15 epochs（约 20-30 分钟，GPU）
4. 在测试集上评估
5. 保存最佳模型和训练信息
"""

import os
import sys

# Windows 编码修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE, NUM_EPOCHS, MODEL_SAVE_PATH
from src.data.dataset import create_data_loaders
from src.models.classifier import create_model
from src.training.trainer import Trainer


def main():
    print("=" * 60)
    print("  胸部X光肺炎检测 — 模型训练")
    print("=" * 60)
    print(f"\n  设备: {DEVICE}")
    if DEVICE.type == "cuda":
        import torch
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. 创建 DataLoader
    train_loader, val_loader, test_loader, class_weights = create_data_loaders()

    # 2. 创建模型
    model = create_model(device=DEVICE)

    # 3. 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
    )

    # 4. 训练
    trainer.train(num_epochs=NUM_EPOCHS)

    # 5. 测试集评估
    test_results = trainer.evaluate_test(test_loader)

    # 6. 总结
    print(f"\n{'=' * 60}")
    print(f"  训练完成！")
    print(f"  最佳模型已保存到: {MODEL_SAVE_PATH}")
    print(f"  测试集准确率: {test_results['accuracy']:.2%}")
    print(f"  测试集 F1-Score: {test_results['f1']:.2%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
