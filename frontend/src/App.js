import React, { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [textResult, setTextResult] = useState(null);

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [imageResult, setImageResult] = useState(null);

  const [cameraResult, setCameraResult] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // ================= TEXT =================
  const handleText = async (value) => {
    setText(value);
    if (value.length < 3) return;

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict-text", {
        text: value,
      });
      setTextResult(res.data);
    } catch {}
  };

  // ================= IMAGE =================
  const handleImageUpload = async (file) => {
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/predict-image",
        formData
      );
      setImageResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const removeImage = () => {
    setImage(null);
    setPreview(null);
    setImageResult(null);
  };

  // ================= CAMERA =================
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert("Camera access denied or not available");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
  };

  // ✅ NEW: REMOVE CAMERA (like image)
  const removeCamera = () => {
    stopCamera();
    setCameraResult(null);

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const captureFrame = async () => {
    if (!videoRef.current || !canvasRef.current) {
      alert("Camera not ready");
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        const res = await axios.post(
          "http://127.0.0.1:8000/predict-image",
          formData
        );
        setCameraResult(res.data);
      } catch (err) {
        console.error(err);
      }
    }, "image/jpeg");
  };

  return (
    <div className="container">
      <h1>🚀 AI Moderation Dashboard</h1>

      <div className="grid">
        {/* TEXT */}
        <div className="card">
          <h2>📝 Text Moderation</h2>

          <textarea
            placeholder="Type here..."
            value={text}
            onChange={(e) => handleText(e.target.value)}
          />

          {textResult && (
            <div className="result">
              <span className={textResult.label}>
                {textResult.label.toUpperCase()}
              </span>
              <p>{(textResult.confidence * 100).toFixed(2)}%</p>
            </div>
          )}
        </div>

        {/* IMAGE */}
        <div className="card">
          <h2>🖼 Image Moderation</h2>

          <label className="upload">
            Upload Image
            <input
              type="file"
              hidden
              onChange={(e) =>
                handleImageUpload(e.target.files[0])
              }
            />
          </label>

          {preview && (
            <img src={preview} className="preview" alt="preview" />
          )}

          {image && (
            <button className="remove" onClick={removeImage}>
              Remove
            </button>
          )}

          {imageResult && (
            <div className="result">
              {imageResult.predictions.map((p, i) => (
                <div key={i}>
                  {p.label} → {(p.score * 100).toFixed(2)}%
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* CAMERA */}
      <div className="card full">
        <h2>📷 Camera Moderation</h2>

        <video ref={videoRef} autoPlay className="video" />
        <canvas ref={canvasRef} style={{ display: "none" }} />

        <div className="buttons">
          <button onClick={startCamera}>Start</button>
          <button onClick={captureFrame}>Analyze</button>
          <button onClick={stopCamera}>Stop</button>

          {/* ✅ NEW REMOVE BUTTON */}
          <button className="remove" onClick={removeCamera}>
            Remove
          </button>
        </div>

        {cameraResult && (
          <div className="result">
            {cameraResult.predictions.map((p, i) => (
              <div key={i}>
                {p.label} → {(p.score * 100).toFixed(2)}%
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;