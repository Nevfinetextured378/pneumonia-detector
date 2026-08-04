"""
数据集下载工具
==============
Kaggle Chest X-Ray Images (Pneumonia) 数据集下载指南。

数据集地址:
    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

下载方式:

1. 手动下载（推荐）:
   - 访问上述链接 → 点击 "Download" 按钮
   - 将下载的 zip 文件解压到: data/chest_xray/
   - 解压后目录结构应为:
     data/chest_xray/
       ├── train/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       ├── val/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       └── test/
           ├── NORMAL/
           └── PNEUMONIA/

2. 使用 Kaggle API（需要先配置 ~/.kaggle/kaggle.json）:
   $ kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   $ unzip chest-xray-pneumonia.zip -d data/chest_xray/
   $ rm chest-xray-pneumonia.zip

验证数据:
    $ python scripts/download_dataset.py --check
"""

import os
import sys

# Windows 编码修复
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR


def check_dataset() -> dict:
    """检查数据集完整性"""
    status = {}

    for name, path in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        normal_dir = os.path.join(path, "NORMAL")
        pneumonia_dir = os.path.join(path, "PNEUMONIA")

        normal_count = len(os.listdir(normal_dir)) if os.path.isdir(normal_dir) else 0
        pneumonia_count = len(os.listdir(pneumonia_dir)) if os.path.isdir(pneumonia_dir) else 0

        status[name] = {
            "path": path,
            "exists": os.path.isdir(path),
            "NORMAL": normal_count,
            "PNEUMONIA": pneumonia_count,
            "total": normal_count + pneumonia_count,
        }

    return status


def print_status(status: dict):
    """打印数据集状态"""
    print("=" * 60)
    print("  胸部X光肺炎数据集 — 状态检查")
    print("=" * 60)

    total_images = 0
    all_ok = True

    for split, info in status.items():
        icon = "✅" if info["exists"] and info["total"] > 0 else "❌"
        print(f"\n{icon} {split.upper()} ({info['path']})")
        if info["exists"]:
            print(f"   NORMAL:    {info['NORMAL']:>6} 张")
            print(f"   PNEUMONIA: {info['PNEUMONIA']:>6} 张")
            print(f"   合计:      {info['total']:>6} 张")
            total_images += info["total"]
        else:
            print(f"   目录不存在")
            all_ok = False

    print(f"\n{'=' * 60}")
    print(f"  总计: {total_images} 张图片")

    if all_ok and total_images > 0:
        print(f"\n✅ 数据集就绪，可以开始训练！")
    else:
        print(f"\n❌ 数据集未就绪，请按照上方说明下载数据集。")
        print(f"   下载地址: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print(f"   解压到:   {DATA_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        print_status(check_dataset())
    else:
        # 打印使用说明
        print(__doc__)
        print("\n--- 当前数据集状态 ---\n")
        print_status(check_dataset())
