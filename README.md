# FastVLM Vision Assistant Demo

A premium, immersive web application that uses the **FastVLM-0.5B-ONNX** model to provide real-time descriptions and answers about what it sees through your camera.

## ✨ Features
- **Immersive UI**: Full-screen camera feed with a modern glassmorphism control panel.
- **Draggable Controls**: Move the interaction panel anywhere on your screen.
- **Edge AI**: Runs entirely in your browser using WebGPU for high performance.
- **Humanized Interaction**: Friendly status updates and real-time responses.

## 🚀 Getting Started

### 1. Prerequisites
- **Browser**: A modern browser with **WebGPU** support (e.g., Chrome, Edge, or Opera).
- **Hardward**: A webcam/camera.
- **Local Server**: Because this app uses ES Modules, it **must** be served from a local server (it won't work by just double-clicking `index.html`).

### 2. Setup & Run
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd demo_NLP
   ```
2. **Start a local server**:
   Use the bundled server so the model can run multi-threaded:
   ```bash
   python serve.py 8000
   ```
   It sends the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy`
   headers that enable `SharedArrayBuffer`. Plain `python -m http.server` works
   too, but forces ONNX Runtime into slow single-threaded mode.
3. **Open the app**:
   Navigate to `http://localhost:8000` in your browser.

## 🛠 Usage
1. **Allow Camera Access**: Click "Allow" when prompted by your browser.
2. **Wait for Loading**: The app will check your computer's speed and "wake up" the AI (downloading the model if it's your first time).
3. **Start Interacting**:
   - Type an instruction in the **Instruction** box (e.g., "What do you see?").
   - Press **Start**.
   - The AI will provide continuous real-time answers based on the camera feed.
   - Press **Stop** to pause the inference.

## 📝 Technical Notes
- **Model**: `onnx-community/FastVLM-0.5B-ONNX`
- **Engine**: Transformers.js (v3+) via CDN.
- **Backend**: WebGPU (optimized with fp16/q4 quantization). Falls back to WASM if WebGPU is unavailable.
