"""
工具函数模块
============
辅助函数：计时、格式化、文件操作等。
"""

import json
import os
import time
from datetime import datetime


def format_time(seconds):
    """将秒数格式化为 00:00:00 格式"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def save_model_info(save_path, info_dict):
    """
    保存模型训练元信息到 JSON 文件。

    Args:
        save_path: JSON 文件保存路径
        info_dict: 包含训练信息的字典
    """
    info_dict["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)

    print(f"\n  模型信息已保存: {save_path}")


class Timer:
    """简单的计时器上下文管理器"""

    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"  {self.name}: {format_time(self.elapsed)}")
