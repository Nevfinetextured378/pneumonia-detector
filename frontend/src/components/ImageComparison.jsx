import React from "react";
import "./ImageComparison.css";

/**
 * 原图 vs Grad-CAM 热力图对比组件
 *
 * @param {string} originalImage - 原图 Base64 data URL
 * @param {string} gradcamImage  - Grad-CAM 叠加图 Base64 data URL
 */
function ImageComparison({ originalImage, gradcamImage }) {
  return (
    <div className="image-comparison">
      <h3 className="comparison-title">影像对比分析</h3>
      <div className="comparison-grid">
        <div className="image-panel">
          <span className="image-badge original-badge">原图 Original</span>
          <img src={originalImage} alt="原始胸部X光片" />
        </div>
        <div className="image-panel">
          <span className="image-badge gradcam-badge">Grad-CAM 热力图</span>
          <img src={gradcamImage} alt="Grad-CAM 可视化热力图" />
        </div>
      </div>
      <p className="comparison-caption">
        Grad-CAM 高亮区域表示模型关注的影像特征，颜色越暖（红/黄）表示该区域对预测结果影响越大。
      </p>
    </div>
  );
}

export default ImageComparison;
