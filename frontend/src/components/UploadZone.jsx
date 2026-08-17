import React, { useState, useRef, useEffect } from "react";
import "./UploadZone.css";

/**
 * 拖拽上传区域
 * - 支持拖拽图片到区域内 / 点击选择文件
 * - 拖入时高亮反馈
 * - 选中后显示预览缩略图
 *
 * @param {string|null}  preview    - 图片预览 URL
 * @param {boolean}      hasFile    - 是否已选择文件
 * @param {boolean}      isLoading  - 是否正在推理
 * @param {function}     onFileSelect - 选中文件回调 (file: File) => void
 * @param {function}     onSubmit   - 提交回调
 */
function UploadZone({ preview, hasFile, isLoading, onFileSelect, onSubmit }) {
  const [isDragActive, setIsDragActive] = useState(false);
  const dragCounter = useRef(0);
  const inputRef = useRef(null);

  // 清理预览 URL（防止内存泄漏）
  useEffect(() => {
    return () => {
      if (preview && preview.startsWith("blob:")) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setIsDragActive(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    dragCounter.current = 0;

    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      onFileSelect(droppedFile);
    }
  };

  const handleInputChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      onFileSelect(selectedFile);
    }
  };

  const handleClickZone = () => {
    if (!isLoading) {
      inputRef.current?.click();
    }
  };

  return (
    <div
      className={`upload-zone${isDragActive ? " drag-active" : ""}${hasFile ? " has-file" : ""}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={handleClickZone}
    >
      {/* 隐藏的文件输入 */}
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/bmp,image/tiff"
        onChange={handleInputChange}
        className="upload-input-hidden"
        tabIndex={-1}
      />

      {!hasFile ? (
        /* ====== 空状态 ====== */
        <div className="upload-empty">
          {/* 上传图标 SVG */}
          <svg className="upload-icon" viewBox="0 0 48 48" width="56" height="56">
            <path
              d="M24 4C18.477 4 14 8.477 14 14c0 .74.083 1.46.236 2.15C9.568 17.822 6 22.464 6 27.889 6 34.314 11.312 39.52 17.777 40H36c4.418 0 8-3.582 8-8 0-3.945-2.855-7.214-6.57-7.874C37.143 22.72 37 21.26 37 19.778 37 15.063 33.19 11.231 28.5 11.033 28.17 7.083 24.5 4 24 4z"
              fill="none"
              stroke="#9ca3af"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <polyline
              points="24,21 24,34"
              fill="none"
              stroke="#9ca3af"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
            <polyline
              points="18,27 24,21 30,27"
              fill="none"
              stroke="#9ca3af"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>

          <p className="upload-hint">
            <strong>拖拽胸部X光图片到此处</strong>
            <br />
            或点击此区域选择文件
          </p>
          <p className="upload-formats">支持 JPEG / PNG / BMP 格式，最大 10MB</p>
        </div>
      ) : (
        /* ====== 已选文件状态 ====== */
        <div className="upload-with-file">
          <div className="preview-wrapper">
            <img src={preview} alt="预览" className="preview-image" />
          </div>
          <div className="upload-actions">
            <button
              type="button"
              className="re-select-btn"
              onClick={(e) => {
                e.stopPropagation();
                inputRef.current?.click();
              }}
              disabled={isLoading}
            >
              重新选择
            </button>
            <button
              type="button"
              className="submit-btn"
              onClick={(e) => {
                e.stopPropagation();
                onSubmit();
              }}
              disabled={isLoading}
            >
              {isLoading ? "检测中..." : "开始检测"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadZone;
