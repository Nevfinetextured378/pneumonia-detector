# 🫁 ChestLens — 胸部X光肺炎AI辅助检测系统

<div align="center">

**基于深度学习的医学影像辅助诊断系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb?logo=react)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

[English](#english) | [中文](#中文)

</div>

---

<a id="english"></a>
## 📖 English

### Overview

**ChestLens** is a deep learning-based computer-aided diagnosis (CAD) system for detecting pneumonia from chest X-ray images. It combines transfer learning with ResNet-50, Grad-CAM visualization, and a modern web interface to provide an end-to-end demo suitable for academic research and educational purposes.

This project was developed as an undergraduate research project at Zhejiang University, Department of Biomedical Engineering.

### Features

- 🖼️ **Image Upload**: Upload chest X-ray images through a web interface
- 🧠 **AI Diagnosis**: Binary classification (Normal / Pneumonia) using fine-tuned ResNet-50
- 🔥 **Grad-CAM Visualization**: Heatmaps showing which image regions influence the model's decision
- 📊 **Probability & Confidence**: Model outputs include prediction probability and confidence scores
- ⚡ **Real-time Inference**: Average inference time < 200ms on GPU
- 🌐 **Modern Web UI**: Clean, responsive React frontend

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User (Browser)                          │
│                  http://localhost:3000                       │
└─────────────────────┬───────────────────────────────────────┘
                      │  HTTP Request (multipart/form-data)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (:8000)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │  POST /api/  │  │  GET /api/   │  │  CORS Middleware  │ │
│  │   predict    │  │   health     │  │                   │ │
│  └──────┬───────┘  └──────────────┘  └───────────────────┘ │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PneumoniaPredictor                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │   │
│  │  │ Preprocess │  │  ResNet-50 │  │   Grad-CAM     │  │   │
│  │  │  (224×224) │  │  Inference │  │  Visualization │  │   │
│  │  └────────────┘  └────────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Deep Learning | PyTorch 2.x, torchvision | Model training & inference |
| Model Architecture | ResNet-50 (ImageNet pretrained) | Transfer learning backbone |
| Visualization | Grad-CAM (custom implementation) | Explainability heatmaps |
| Backend | FastAPI + Uvicorn | REST API server |
| Frontend | React 18 + Vite | User interface |
| Data Processing | NumPy, Pandas, OpenCV, Pillow | Image I/O & preprocessing |
| Evaluation | scikit-learn | Metrics (Accuracy, Precision, Recall, F1) |
| Notebook | Jupyter | Data exploration & analysis |

### Project Structure

```
pneumonia-detector/
├── app/                        # FastAPI 后端应用
│   ├── main.py                 #   应用入口 + 生命周期管理
│   └── routers/
│       └── predict.py          #   /api/predict 路由
├── frontend/                   # React 前端
│   ├── src/
│   │   ├── App.jsx             #   主组件（上传 + 结果展示）
│   │   ├── App.css             #   样式
│   │   └── index.jsx           #   入口
│   ├── index.html              #   HTML 模板
│   └── vite.config.js          #   Vite 配置
├── src/                        # 核心 Python 库
│   ├── config.py               #   全局配置（路径、超参数）
│   ├── data/
│   │   └── dataset.py          #   数据集加载 + 数据增强
│   ├── models/
│   │   └── classifier.py       #   ResNet-50 迁移学习模型
│   ├── training/
│   │   └── trainer.py          #   训练循环 + 评估
│   ├── inference/
│   │   ├── predictor.py        #   推理引擎
│   │   └── gradcam.py          #   Grad-CAM 热力图
│   └── utils/
│       └── helpers.py          #   工具函数
├── scripts/
│   ├── train.py                #   训练入口脚本
│   └── download_dataset.py     #   数据集下载 & 检查工具
├── notebooks/
│   └── 01_data_exploration.ipynb  # 数据探索 Notebook
├── models/
│   ├── best_model.pth          #   最佳模型权重（训练后生成）
│   └── model_info.json         #   训练元信息
├── data/chest_xray/            #   数据集（需自行下载）
├── pyproject.toml              #   项目配置 & 依赖
├── README.md                   #   本文件
└── LICENSE                     #   MIT 许可证
```

### Quick Start

#### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- (Optional) NVIDIA GPU with CUDA for faster training/inference

#### 1. Clone & Setup Environment

```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-detector.git
cd pneumonia-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -e .
```

#### 2. Download Dataset

```bash
# Check dataset status
python scripts/download_dataset.py --check

# Download manually from Kaggle:
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Extract to: data/chest_xray/
```

#### 3. Train the Model

```bash
# Train ResNet-50 with transfer learning (~25 min on GPU)
python scripts/train.py
```

#### 4. Start the Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

#### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

#### 6. Use the System

1. Open the web interface
2. Upload a chest X-ray image (JPEG/PNG)
3. Click "开始检测"
4. View the prediction result and Grad-CAM heatmap

### API Reference

#### `POST /api/predict`

Upload a chest X-ray image for pneumonia detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (JPEG/PNG, max 10MB)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "PNEUMONIA",
    "class_cn": "肺炎",
    "probability": 0.9234,
    "confidence": 92.34
  },
  "probabilities": {
    "NORMAL": 0.0766,
    "PNEUMONIA": 0.9234
  },
  "gradcam_image": "data:image/png;base64,iVBORw0KGgo...",
  "original_image": "data:image/png;base64,iVBORw0KGgo...",
  "inference_time_ms": 145.32
}
```

#### `GET /api/health`

Health check endpoint.

### Model Performance

| Metric | Value |
|--------|-------|
| Model | ResNet-50 (ImageNet pretrained) |
| Training Data | 5,216 chest X-ray images |
| Best Validation Accuracy | **88.48%** |
| Test Accuracy | Varies (see training output) |
| Training Time | ~25 minutes (NVIDIA GPU) |
| Inference Time | ~150ms per image (GPU) |

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney

### Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights regions of the input image that most influenced the model's prediction. This provides interpretability for the AI's decision-making process.

- **Red/Yellow areas**: High activation — regions the model focuses on
- **Blue areas**: Low activation — regions with less influence

### Roadmap

- [x] ResNet-50 transfer learning baseline
- [x] Grad-CAM heatmap visualization
- [x] FastAPI backend with REST API
- [x] React frontend with image upload
- [ ] Docker containerization
- [ ] Model ensemble (ResNet + DenseNet + EfficientNet)
- [ ] Multi-class classification (bacterial vs viral pneumonia)
- [ ] Patient report generation (PDF)
- [ ] Deployment to cloud (HuggingFace Spaces / Railway)

### ⚠️ Disclaimer

**This tool is for RESEARCH and EDUCATIONAL purposes only.** It is NOT approved for clinical use and should NOT be used for actual medical diagnosis. The model's predictions are not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

### Acknowledgments

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016

### License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

<a id="中文"></a>
## 📖 中文

### 项目简介

**ChestLens（胸部X光肺炎AI辅助检测系统）** 是一个基于深度学习的医学影像辅助诊断（CAD）系统。使用迁移学习（ResNet-50）、Grad-CAM 可视化和现代 Web 技术，实现从图片上传到 AI 诊断的完整流程演示。

本项目为浙江大学生物医学工程专业本科生科研训练项目。

### 功能特性

- 🖼️ **图片上传**：通过网页界面上传胸部X光片
- 🧠 **AI 诊断**：使用微调的 ResNet-50 进行正常/肺炎二分类
- 🔥 **Grad-CAM 可视化**：热力图展示 AI 关注的影像区域
- 📊 **概率与置信度**：输出包含疾病概率和模型置信度
- ⚡ **实时推理**：GPU 下单张推理 < 200ms
- 🌐 **现代 Web 界面**：简洁的 React 前端

### 系统架构

```
用户浏览器 (:3000)
    │  HTTP 请求（上传图片）
    ▼
FastAPI 后端 (:8000)
    ├── POST /api/predict    ← 图片预测 + Grad-CAM
    ├── GET  /api/health     ← 健康检查
    └── PneumoniaPredictor
        ├── 图像预处理 (224×224, ImageNet归一化)
        ├── ResNet-50 推理
        └── Grad-CAM 热力图生成
```

### 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 深度学习 | PyTorch 2.x + torchvision | 模型训练与推理 |
| 模型架构 | ResNet-50（ImageNet 预训练） | 迁移学习骨干网络 |
| 可解释性 | Grad-CAM（自实现） | 热力图可视化 |
| 后端 | FastAPI + Uvicorn | REST API 服务 |
| 前端 | React 18 + Vite | 用户界面 |
| 数据处理 | NumPy, OpenCV, Pillow | 图像处理 |
| 评估 | scikit-learn | 准确率/精确率/召回率/F1 |

### 快速开始

#### 环境要求

- Python 3.10+
- Node.js 18+
- （可选）NVIDIA GPU + CUDA

#### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/pneumonia-detector.git
cd pneumonia-detector

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -e .
```

#### 2. 下载数据集

```bash
# 检查数据集状态
python scripts/download_dataset.py --check

# 从 Kaggle 手动下载：
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# 解压到: data/chest_xray/
```

#### 3. 训练模型

```bash
python scripts/train.py
# GPU 约 25 分钟，CPU 约 2-3 小时
```

#### 4. 启动后端

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# API 文档: http://localhost:8000/docs
```

#### 5. 启动前端

```bash
cd frontend
npm install
npm run dev
# 打开: http://localhost:3000
```

#### 6. 使用系统

1. 打开网页 http://localhost:3000
2. 选择一张胸部X光图片
3. 点击「开始检测」
4. 查看预测结果和 Grad-CAM 热力图

### 模型性能

| 指标 | 数值 |
|------|------|
| 模型 | ResNet-50（ImageNet 预训练） |
| 训练数据 | 5,216 张胸部X光片 |
| 最佳验证准确率 | **88.48%** |
| 训练耗时 | ~25 分钟（GPU） |
| 推理耗时 | ~150ms/张（GPU） |

**数据集**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### 项目结构

参见上方 [Project Structure](#project-structure) 部分。

### 后续计划

- [x] ResNet-50 迁移学习基线
- [x] Grad-CAM 热力图可视化
- [x] FastAPI 后端 API
- [x] React 前端界面
- [ ] Docker 容器化部署
- [ ] 多模型集成（ResNet + DenseNet）
- [ ] 细菌性/病毒性肺炎多分类
- [ ] 云端部署（HuggingFace Spaces）

### ⚠️ 免责声明

**本工具仅供科研和教育用途。** 不得用于临床诊断。AI 模型的预测结果不能替代专业医生的诊断意见。如有健康问题，请咨询专业医疗机构。

### 参考资料

- 📄 **Grad-CAM 论文**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- 📄 **ResNet 论文**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- 📊 **数据集**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### 联系作者

- 作者：浙江大学 生物医学工程专业
- 项目性质：本科生科研训练项目

---

<div align="center">
  <sub>Built with ❤️ for better healthcare AI education</sub>
</div>
