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
import time
import wave
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
# es_MX-ald is neutral Latin American, which sits much closer to Colombian ears
# than the Castilian voices. Piper has no es_CO voice; es_AR-daniela is the other
# Latin American option.
VOICE_PATH = Path(os.environ.get(
    "TTS_VOICE",
    f"/disk/{os.environ.get('USER', '')}/piper/es_MX-ald-medium.onnx",
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

# Loaded once, here. Shelling out to the piper CLI per sentence meant loading the
# ONNX model on every request, which was most of the wait.
voice = None
if VOICE_PATH.exists():
    try:
        from piper import PiperVoice

        print(f"loading piper voice {VOICE_PATH.name}...", flush=True)
        voice = PiperVoice.load(str(VOICE_PATH))
        print("piper ready", flush=True)
    except Exception as e:      # noqa: BLE001 - any failure here just means no TTS
        print(f"piper unavailable: {e}", flush=True)
else:
    print(f"no piper voice at {VOICE_PATH}, /speak will return 503", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": "cpu",
        "voice": VOICE_PATH.name if voice is not None else None,
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


@app.post("/speak")
def speak(req: SpeakRequest):
    if voice is None:
        raise HTTPException(503, f"No hay voz Piper cargada ({VOICE_PATH})")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "texto vacío")

    started = time.perf_counter()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        # synthesize_wav is the current name; older piper-tts releases called it
        # synthesize, and both write a complete WAV into the file object.
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wav)
        else:
            voice.synthesize(text, wav)

    audio = buf.getvalue()
    elapsed = time.perf_counter() - started
    print(f"[tts] {elapsed:.2f}s  {len(text)} chars -> {len(audio) // 1024}KB", flush=True)

    return Response(content=audio, media_type="audio/wav")
