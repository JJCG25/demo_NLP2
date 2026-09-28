#!/usr/bin/env python3
"""Local speech-to-text, so no audio ever leaves the machine.

Chrome's SpeechRecognition streams the microphone to Google. This replaces it
with Whisper running on the workstation's CPUs, which sit idle while the four
T4s are busy with the vision model.

    uvicorn cluster.stt_server:app --host 0.0.0.0 --port 9000

The sbatch starts it alongside vLLM. CPU on purpose: the GPUs are full, and
`small` on int8 transcribes a few seconds of speech in about the same time it
takes to say it.
"""
import io
import os
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# `small` is the sweet spot for Spanish on CPU: `base` mishears names and
# technical words, `medium` roughly triples the wait.
MODEL_SIZE = os.environ.get("STT_MODEL", "small")
LANGUAGE = os.environ.get("STT_LANG", "es")
THREADS = int(os.environ.get("STT_THREADS", "8"))

app = FastAPI()

# The page is served from localhost:8080 and reaches this through a tunnel on
# another port, so the browser treats it as cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"loading whisper {MODEL_SIZE} on cpu ({THREADS} threads)...", flush=True)
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=THREADS)
print("whisper ready", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE, "device": "cpu"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    started = time.perf_counter()
    raw = await audio.read()

    # faster-whisper decodes the container itself (PyAV), so the browser's
    # webm/opus blob goes in as it arrives.
    segments, info = model.transcribe(
        io.BytesIO(raw),
        language=LANGUAGE,
        beam_size=1,           # greedy: this is short speech, and latency wins
        vad_filter=True,       # drops silence and room noise before decoding
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()

    elapsed = time.perf_counter() - started
    print(f"[stt] {elapsed:.2f}s  {len(raw) // 1024}KB  -> {text!r}", flush=True)

    return {"text": text, "seconds": round(elapsed, 2)}
