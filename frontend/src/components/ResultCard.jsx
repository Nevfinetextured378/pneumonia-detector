import React from "react";
import ProbabilityGauge from "./ProbabilityGauge";
import ImageComparison from "./ImageComparison";
import "./ResultCard.css";

/**
 * 诊断结果卡片
 * 包含：诊断横幅、概率仪表盘、统计磁贴、影像对比、重新检测按钮
 *
 * @param {object}   result   - API 返回的预测结果
 * @param {function} onReset  - 重置回调（重新检测）
 */
function ResultCard({ result, onReset }) {
  const isPneumonia = result.prediction.class === "PNEUMONIA";

  const badgeInfo = isPneumonia
    ? {
        icon: "!",
        title: "检测到肺炎迹象",
        enTitle: "Pneumonia Detected",
        subtitle: "建议进一步临床评估",
        type: "positive",
      }
    : {
        icon: "✓",
        title: "未见明显肺炎特征",
        enTitle: "No Pneumonia Detected",
        subtitle: "肺部影像未见明显异常",
        type: "negative",
      };

  return (
    <section className="result-card">
      {/* ===== A) 诊断横幅 ===== */}
      <div className={`diagnosis-banner ${badgeInfo.type}`}>
        <span className={`banner-icon ${badgeInfo.type}`}>{badgeInfo.icon}</span>
        <div className="banner-text">
          <span className="banner-title">{badgeInfo.title}</span>
          <span className="banner-subtitle">
            {badgeInfo.enTitle} — {badgeInfo.subtitle}
          </span>
        </div>
      </div>

      {/* ===== B) 概率仪表盘 ===== */}
      <div className="gauge-wrapper">
        <ProbabilityGauge
          probability={result.prediction.probability}
          isPneumonia={isPneumonia}
        />
      </div>

      {/* ===== C) 统计磁贴 ===== */}
      <div className="stat-tiles">
        <div className="stat-tile">
          <span className="stat-label">置信度 Confidence</span>
          <span className="stat-value">{result.prediction.confidence.toFixed(1)}%</span>
        </div>
        <div className="stat-tile">
          <span className="stat-label">推理时间 Inference</span>
          <span className="stat-value">{result.inference_time_ms} ms</span>
        </div>
      </div>

      {/* ===== D) 影像对比 ===== */}
      <ImageComparison
        originalImage={result.original_image}
        gradcamImage={result.gradcam_image}
      />

      {/* ===== E) 重新检测按钮 ===== */}
      <div className="reset-wrapper">
        <button className="reset-btn" onClick={onReset}>
          ← 重新检测
        </button>
      </div>
    </section>
  );
}

export default ResultCard;
