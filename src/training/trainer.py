"""
模型训练模块
============
提供模型训练、验证、测试的完整流程。

训练策略：
- 损失函数: CrossEntropyLoss（带类别权重处理不平衡）
- 优化器: Adam（比 SGD 对学习率更鲁棒，适合初学者）
- 学习率调度: StepLR（每 LR_STEP_SIZE 个 epoch 减半）
- 模型保存: 每个 epoch 保存 checkpoint，同时跟踪最佳模型
"""

import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 修复 Windows 编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    LR_STEP_SIZE,
    LR_GAMMA,
    MODEL_SAVE_PATH,
    MODEL_INFO_PATH,
    CLASS_NAMES,
)
from src.utils.helpers import save_model_info, format_time


class Trainer:
    """
    模型训练器。

    Args:
        model: PyTorch 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        class_weights: 类别权重（用于不平衡数据处理）
        learning_rate: 学习率
        weight_decay: 权重衰减（L2 正则化）
        device: 训练设备
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        class_weights=None,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 损失函数（带类别权重）
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 优化器：只优化需要梯度的参数
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
        )

        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        print(f"\n  Trainer 初始化完成")
        print(f"  优化器: Adam (lr={learning_rate}, wd={weight_decay})")
        print(f"  学习率调度: StepLR (step={LR_STEP_SIZE}, gamma={LR_GAMMA})")

    def train_epoch(self):
        """执行一个 epoch 的训练"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 进度条
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(self.train_loader) - 1:
                progress = (batch_idx + 1) / len(self.train_loader) * 100
                print(
                    f"\r    训练进度: {progress:3.0f}% "
                    f"[{'=' * int(progress // 5)}{' ' * (20 - int(progress // 5))}] "
                    f"Loss: {loss.item():.4f}",
                    end="",
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        print(f"\r    训练完成: Loss={avg_loss:.4f}, Acc={accuracy:.2%}")

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, loader=None, desc="验证"):
        """在给定数据加载器上评估模型"""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy, all_preds, all_labels

    def train(self, num_epochs=NUM_EPOCHS):
        """
        完整训练流程。

        Args:
            num_epochs: 训练总轮数
        """
        print("\n" + "=" * 60)
        print(f"  开始训练 ({num_epochs} epochs)")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\n  ┌─ Epoch {epoch}/{num_epochs} (lr={current_lr:.2e}) ─┐")

            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # 验证
            val_loss, val_acc, _, _ = self.validate(desc="验证")
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            print(f"    验证: Loss={val_loss:.4f}, Acc={val_acc:.2%}")

            # 更新学习率
            self.scheduler.step()

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint(is_best=True)
                print(f"    ★ 新最佳模型！ (Acc: {val_acc:.2%})")
            else:
                # 每 5 个 epoch 保存一次常规 checkpoint
                if epoch % 5 == 0:
                    self._save_checkpoint(is_best=False)

        total_time = time.time() - start_time
        print(f"\n  {'=' * 60}")
        print(f"  训练完成！总耗时: {format_time(total_time)}")
        print(f"  最佳验证准确率: {self.best_val_acc:.2%} (Epoch {self.best_epoch})")
        print(f"  {'=' * 60}")

        # 保存模型信息
        save_model_info(MODEL_INFO_PATH, self._get_info_dict(total_time))

    def _save_checkpoint(self, is_best=False):
        """保存模型 checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

        if is_best:
            torch.save(checkpoint, MODEL_SAVE_PATH)
        else:
            epoch_path = os.path.join(
                os.path.dirname(MODEL_SAVE_PATH),
                f"checkpoint_epoch_{self.current_epoch}.pth",
            )
            torch.save(checkpoint, epoch_path)

    def _get_info_dict(self, total_time):
        """生成模型训练信息字典"""
        return {
            "model_name": "resnet50",
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "best_val_accuracy": float(self.best_val_acc),
            "best_epoch": self.best_epoch,
            "total_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": self.train_loader.batch_size,
            "training_time_seconds": total_time,
            "device": str(self.device),
            "final_train_loss": float(self.train_losses[-1]) if self.train_losses else None,
            "final_val_loss": float(self.val_losses[-1]) if self.val_losses else None,
        }

    def evaluate_test(self, test_loader):
        """
        在测试集上评估并打印详细报告。

        Args:
            test_loader: 测试数据加载器
        """
        print("\n" + "=" * 60)
        print("  测试集评估")
        print("=" * 60)

        test_loss, test_acc, all_preds, all_labels = self.validate(
            test_loader, desc="测试"
        )

        # 计算各指标
        precision = precision_score(all_labels, all_preds, average="binary")
        recall = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n  测试准确率: {test_acc:.2%}")
        print(f"  精确率 (Precision): {precision:.2%}")
        print(f"  召回率 (Recall):    {recall:.2%}")
        print(f"  F1-Score:           {f1:.2%}")
        print(f"\n  混淆矩阵:")
        print(f"                预测")
        print(f"              NORMAL  PNEUMONIA")
        print(f"  实际 NORMAL    {cm[0][0]:4d}     {cm[0][1]:4d}")
        print(f"  实际 PNEUMONIA {cm[1][0]:4d}     {cm[1][1]:4d}")

        print("=" * 60)

        return {
            "accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        }
