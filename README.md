# Qwen2.5-VL Vision Assistant Demo

A web application that answers questions about what it sees through your camera,
in Spanish, using **Qwen2.5-VL-32B-Instruct-AWQ** served by vLLM on 4x Tesla T4.

The browser holds the camera and the UI. The model runs on the GPU box and is
reached over an SSH tunnel, so the page only ever talks to `localhost`.

```
laptop                              T4 box
------                              ------
camera + UI  --- ssh tunnel --->    vLLM (4x T4, TP=4)
                :8000               Qwen2.5-VL-32B-AWQ
```

## ✨ Features
- **Immersive UI**: Full-screen camera feed with a modern glassmorphism control panel.
- **Draggable Controls**: Move the interaction panel anywhere on your screen.
- **Spanish answers**: The system prompt pins the response language.
- **Timing readout**: Each answer reports its latency, and the console logs
  tokens and frame size so you can tune.

## 🚀 Getting Started

### 1. Prerequisites
- **GPU box**: 4x T4 (64 GB total) running Linux, with `pip install vllm`.
  vLLM has no native Windows build; on Windows use WSL2 or Docker.
- **Laptop**: any modern browser and a webcam. No WebGPU needed any more.
- **SSH access** from the laptop to the GPU box.

### 2. Start the model (on the T4 box)
```bash
./run_vllm.sh
```
First start downloads ~20 GB of int4 weights. Wait for `Application startup complete`.
See the comments in [run_vllm.sh](run_vllm.sh) for the T4-specific flags and what
to do about OOM.

### 3. Open the tunnel (on the laptop)
```bash
ssh -L 8000:localhost:8000 user@t4-box
```
Leave it running. This is why no CORS setup or exposed port is needed.

### 4. Serve the page (on the laptop)
```bash
python serve.py 8080
```
Then open `http://localhost:8080`. Port 8080, not 8000 — the tunnel owns 8000.

## 🛠 Usage
1. **Allow Camera Access**: Click "Allow" when prompted by your browser.
2. **Check the status line**: It should read `✅ Ready!`. If it can't reach the
   server it tells you so, and the tunnel is the usual cause.
3. **Ask something**: Type an instruction (e.g. "¿Qué hay sobre la mesa?") and
   press **Analyze**. One frame is captured and analyzed per press.

## ⚡ Tuning latency
Expect a few seconds per answer. T4s are pre-Ampere, so there is no
FlashAttention-2 and no bf16, and a 32B model is genuinely heavy. In order of
impact:

| Knob | Where | Effect |
|---|---|---|
| `MAX_SIDE` | [script.js](script.js) | Frame resolution. Fewer pixels, fewer vision tokens, faster prefill. Keep `max_pixels` in `run_vllm.sh` in sync. |
| `MAX_TOKENS` | [script.js](script.js) | Caps answer length. Decoding is token-by-token, so this scales latency directly. |
| `--tensor-parallel-size` | [run_vllm.sh](run_vllm.sh) | 4 spreads compute but costs PCIe traffic (no NVLink on T4). Measure 2 against 4. |

If it's still too slow, `Qwen/Qwen2.5-VL-7B-Instruct` in fp16 fits on a single
T4 and is several times faster. Only `MODEL_ID` and the `vllm serve` argument change.

## 📝 Technical Notes
- **Model**: [`Qwen/Qwen2.5-VL-32B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ) (int4, ~20 GB)
- **Engine**: vLLM, OpenAI-compatible `/v1/chat/completions`
- **Precision**: `float16`, required — T4 (sm75) has no bfloat16 support
- **Why not Qwen3-VL**: it has [no vLLM backend for Turing GPUs](https://github.com/vllm-project/vllm/issues/29743)
- **`serve.py`**: the COOP/COEP headers it sends are harmless leftovers from the
  in-browser era; `python -m http.server 8080` now works just as well.
