import React, { useState } from "react";
import UploadZone from "./components/UploadZone";
import LoadingSpinner from "./components/LoadingSpinner";
import ResultCard from "./components/ResultCard";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // ---- 处理文件选择 ----
  const handleFileSelect = (selectedFile) => {
    // 清理旧数据
    setError(null);
    setResult(null);
    if (preview && preview.startsWith("blob:")) {
      URL.revokeObjectURL(preview);
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  // ---- 提交检测 ----
  const handleSubmit = async () => {
    if (!file) {
      setError("请先选择一张胸部X光图片。");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "上传失败，请重试。");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ---- 重置（重新检测）----
  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (preview && preview.startsWith("blob:")) {
      URL.revokeObjectURL(preview);
    }
  };

  return (
    <div className="app-container">
      {/* ===== Header ===== */}
      <header className="app-header">
        <div className="header-accent" />
        <h1 className="header-title">胸部X光肺炎AI辅助检测</h1>
        <p className="header-desc">
          基于 ResNet-50 迁移学习 + Grad-CAM 可解释性可视化
        </p>
      </header>

      {/* ===== Main ===== */}
      <main>
        {/* 上传卡片（含加载叠加层） */}
        <section className="upload-card">
          <UploadZone
            preview={preview}
            hasFile={!!file}
            isLoading={loading}
            onFileSelect={handleFileSelect}
            onSubmit={handleSubmit}
          />
          <LoadingSpinner isVisible={loading} />
        </section>

        {/* 错误提示 */}
        {error && !loading && (
          <div className="error-banner">
            <span className="error-icon">⚠</span>
            <span>{error}</span>
            <button className="error-close" onClick={() => setError(null)}>×</button>
          </div>
        )}

        {/* 诊断结果 */}
        {result && !loading && (
          <ResultCard result={result} onReset={handleReset} />
        )}
      </main>

      {/* ===== Footer ===== */}
      <footer className="app-footer">
        <span>浙江大学 · 生物医学工程</span>
        <span className="footer-sep">|</span>
        <span>科研教育项目 · 不适用于临床诊断</span>
      </footer>
    </div>
  );
}

export default App;
