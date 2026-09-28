#!/usr/bin/env python3
"""Local speech, both directions, so no audio ever leaves the machine.

Chrome's SpeechRecognition streams the microphone to Google, and its good-sounding
voices synthesise on Google's servers too. This replaces both with models on the
workstation's CPUs, which sit idle while the four T4s hold the vision model:

    POST /transcribe   audio  -> text   (Whisper via faster-whisper)
    POST /speak        text   -> WAV    (Piper)

    uvicorn cluster.stt_server:app --host 0.0.0.0 --port 9000

The sbatch starts it alongside vLLM. CPU on purpose: the GPUs are full, and both
models are small enough to keep up with speech.
"""
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from faster_whisper import WhisperModel
from pydantic import BaseModel

# `small` is the sweet spot for Spanish on CPU: `base` mishears names and
# technical words, `medium` roughly triples the wait.
MODEL_SIZE = os.environ.get("STT_MODEL", "small")
LANGUAGE = os.environ.get("STT_LANG", "es")
THREADS = int(os.environ.get("STT_THREADS", "8"))

# Piper voice for the answers. setup_env.sh downloads it; without it /speak says
# 503 and the browser falls back to whatever voice the laptop has.
VOICE_PATH = Path(os.environ.get(
    "TTS_VOICE",
    f"/disk/{os.environ.get('USER', '')}/piper/es_ES-davefx-medium.onnx",
)).expanduser()

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
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": "cpu",
        "voice": VOICE_PATH.name if VOICE_PATH.exists() else None,
    }


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


class SpeakRequest(BaseModel):
    text: str


def piper_command():
    """The console script when pip installed it, the module otherwise: which one
    exists has moved between piper-tts releases."""
    exe = shutil.which("piper")
    return [exe] if exe else [sys.executable, "-m", "piper"]


@app.post("/speak")
def speak(req: SpeakRequest):
    if not VOICE_PATH.exists():
        raise HTTPException(503, f"No hay voz Piper en {VOICE_PATH}")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "texto vacío")

    started = time.perf_counter()
    # One sentence at a time, so speech starts while the model is still writing.
    # "-" writes the WAV to stdout.
    proc = subprocess.run(
        piper_command() + ["--model", str(VOICE_PATH), "--output_file", "-"],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise HTTPException(500, proc.stderr.decode("utf-8", "replace")[-400:])

    elapsed = time.perf_counter() - started
    print(f"[tts] {elapsed:.2f}s  {len(text)} chars -> {len(proc.stdout) // 1024}KB", flush=True)

    return Response(content=proc.stdout, media_type="audio/wav")
