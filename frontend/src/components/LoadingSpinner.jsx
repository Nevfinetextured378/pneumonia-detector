import React from "react";
import "./LoadingSpinner.css";

/**
 * 医疗风加载动画
 * - 脉冲扫描波纹
 * - 旋转十字图标
 * - 动态省略号文字
 *
 * @param {boolean} isVisible - 是否显示
 */
function LoadingSpinner({ isVisible }) {
  if (!isVisible) return null;

  return (
    <div className="loading-overlay">
      <div className="spinner-container">
        {/* 脉冲外环 */}
        <div className="spinner-ring" />
        {/* 旋转十字 */}
        <div className="spinner-cross">+</div>
      </div>
      <p className="loading-text">
        AI 正在分析影像
        <span className="loading-dots">
          <span className="dot">.</span>
          <span className="dot">.</span>
          <span className="dot">.</span>
        </span>
      </p>
    </div>
  );
}

export default LoadingSpinner;
