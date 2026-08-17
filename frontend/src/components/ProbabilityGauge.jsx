import React, { useId } from "react";
import "./ProbabilityGauge.css";

/**
 * SVG 环形概率仪表盘
 *
 * @param {number} probability  - 0~1 之间的概率值
 * @param {boolean} isPneumonia - 是否为肺炎（决定颜色主题）
 */
function ProbabilityGauge({ probability, isPneumonia }) {
  const gradientId = useId();
  const radius = 60;
  const circumference = 2 * Math.PI * radius; // ≈ 376.99
  const offset = circumference * (1 - probability);

  const colors = isPneumonia
    ? { start: "#f97316", end: "#ef4444" }   // 橙 → 红
    : { start: "#10b981", end: "#059669" };   // 绿 → 青

  return (
    <div className="gauge-container">
      <svg viewBox="0 0 140 140" className="gauge-svg">
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={colors.start} />
            <stop offset="100%" stopColor={colors.end} />
          </linearGradient>
        </defs>

        {/* 背景轨道 */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="10"
        />

        {/* 前景进度弧 */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 70 70)"
          className="gauge-arc"
        />

        {/* 中心百分比 */}
        <text x="70" y="63" textAnchor="middle" fontSize="28" fontWeight="700" fill="#111827">
          {(probability * 100).toFixed(1)}%
        </text>

        {/* 中心标签 */}
        <text x="70" y="88" textAnchor="middle" fontSize="12" fill="#6b7280">
          {isPneumonia ? "肺炎概率" : "正常概率"}
        </text>
      </svg>
    </div>
  );
}

export default ProbabilityGauge;
