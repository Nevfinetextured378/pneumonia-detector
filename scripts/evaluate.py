#!/usr/bin/env python
"""
测试集评估脚本
==============
在独立测试集上评估已训练的模型，输出完整的分类指标、
混淆矩阵与 ROC 曲线，结果保存到 results/ 目录。

命令行运行:
    python scripts/evaluate.py

产物:
    results/evaluation_report.json   # 指标 JSON（供 README / 报告引用）
    results/confusion_matrix.png     # 混淆矩阵图
    results/roc_curve.png            # ROC 曲线
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # 无界面后端，避免在无显示环境报错
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# 修复 Windows 编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PROJECT_ROOT,
    TEST_DIR,
    MODEL_SAVE_PATH,
    DEVICE,
    CLASS_NAMES,
    BATCH_SIZE,
    NUM_WORKERS,
)
from src.data.dataset import ChestXrayDataset, get_val_transform
from src.models.classifier import PneumoniaClassifier

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def load_model(model_path=MODEL_SAVE_PATH):
    """加载训练好的模型权重并切换到评估模式"""
    model = PneumoniaClassifier()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, loader):
    """对测试集全部样本推理，返回 (标签, 预测, 肺炎概率)"""
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)  # [B, 2]

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # 肺炎类（索引 1）的概率

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 标注格子数字（根据背景深浅自动切换文字颜色）
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=20,
            )

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (ResNet-50, Test Set)")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(fpr, tpr, auc, save_path):
    """绘制 ROC 曲线"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr, tpr, color="tab:blue", lw=2, label=f"ResNet-50 (AUC = {auc:.4f})"
    )
    ax.plot(
        [0, 1], [0, 1], color="gray", lw=1, linestyle="--",
        label="Random (AUC = 0.5)",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (PNEUMONIA as Positive Class)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  测试集评估 — ResNet-50 肺炎检测")
    print("=" * 60)
    print(f"\n  设备: {DEVICE}")
    print(f"  模型: {MODEL_SAVE_PATH}")

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    model = load_model()

    # 2. 构建测试集 DataLoader（无数据增强，与验证集一致）
    print("[2/4] 加载测试集...")
    test_dataset = ChestXrayDataset(TEST_DIR, transform=get_val_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )
    print(f"  测试集样本数: {len(test_dataset)}")
    counts = test_dataset.get_class_counts()
    print(f"  类别分布: {counts}")

    # 3. 推理
    print("[3/4] 推理中...")
    y_true, y_pred, y_probs = predict_all(model, test_loader)

    # 4. 计算指标
    print("[4/4] 计算指标...\n")

    accuracy = accuracy_score(y_true, y_pred)
    # 正类 = PNEUMONIA（索引 1），即"肺炎"为阳性
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    # 各类别详细指标
    per_class_report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4
    )

    # ---- 打印报告 ----
    print("-" * 60)
    print("  评估结果")
    print("-" * 60)
    print(f"  准确率 (Accuracy)         : {accuracy:.4f} ({accuracy:.2%})")
    print(f"  精确率 (Precision, 肺炎)  : {precision:.4f}")
    print(f"  召回率 (Recall, 肺炎)     : {recall:.4f}")
    print(f"  F1-Score (肺炎)           : {f1:.4f}")
    print(f"  ROC-AUC                   : {auc:.4f}")
    print()
    print("  混淆矩阵 (行=实际, 列=预测):")
    print(f"                 预测NORMAL  预测PNEUMONIA")
    print(f"  实际NORMAL         {cm[0, 0]:4d}          {cm[0, 1]:4d}")
    print(f"  实际PNEUMONIA      {cm[1, 0]:4d}          {cm[1, 1]:4d}")
    print()
    print("  分类报告:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # ---- 保存结果 ----
    report = {
        "model": "resnet50",
        "checkpoint": os.path.relpath(MODEL_SAVE_PATH, PROJECT_ROOT),
        "device": str(DEVICE),
        "test_size": len(test_dataset),
        "class_distribution": counts,
        "metrics": {
            "accuracy": float(accuracy),
            "precision_pneumonia": float(precision),
            "recall_pneumonia": float(recall),
            "f1_pneumonia": float(f1),
            "roc_auc": float(auc),
        },
        "per_class": {
            name: {
                "precision": float(per_class_report[name]["precision"]),
                "recall": float(per_class_report[name]["recall"]),
                "f1": float(per_class_report[name]["f1-score"]),
                "support": int(per_class_report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": cm.tolist(),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = os.path.join(RESULTS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 绘制并保存图
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plot_confusion_matrix(cm, cm_path)
    plot_roc_curve(fpr, tpr, auc, roc_path)

    print()
    print("=" * 60)
    print("  结果已保存:")
    print(f"    报告:      {report_path}")
    print(f"    混淆矩阵:  {cm_path}")
    print(f"    ROC 曲线:  {roc_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
