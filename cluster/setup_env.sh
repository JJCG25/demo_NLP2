#!/usr/bin/env bash
# One-time setup. Run this ON THE LOGIN NODE, not inside a job.
#
#   bash cluster/setup_env.sh
#
# Everything lands on /disk: the venv, the HF cache and the ~20GB of weights.
# The default HF cache is ~/.cache/huggingface, which would eat your home quota.
#
# It also pre-downloads the weights here, on the login node, because compute
# nodes on many clusters have no internet access. If yours do, this step just
# warms the cache anyway.
set -euo pipefail

# --- EDIT THIS if your space is somewhere else ---
DISK_ROOT="/disk/$USER"
# -------------------------------------------------

MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

export HF_HOME="$DISK_ROOT/hf"

# Every cache off $HOME, which is full. The pip one matters as much as the
# weights: installing vllm pulls a CUDA build of torch, and pip caches the
# wheels in ~/.cache/pip on the way, which is several GB by itself.
export PIP_CACHE_DIR="$DISK_ROOT/cache/pip"
export XDG_CACHE_HOME="$DISK_ROOT/cache"

mkdir -p "$DISK_ROOT"/{hf,logs,vllm-cache,cache}

# Slurm writes job logs into the directory you submit from, so a repo living in
# $HOME would put them there too.
case "$PWD" in
    /disk/*) ;;
    *) echo "WARNING: you are in $PWD, not on /disk."
       echo "         Clone the repo under $DISK_ROOT so job logs land there too."
       echo ;;
esac

echo "==> Space available on $DISK_ROOT (need ~25GB free):"
df -h "$DISK_ROOT" | tail -1

if [ ! -d "$DISK_ROOT/venv" ]; then
    echo "==> Creating venv at $DISK_ROOT/venv"
    python3 -m venv "$DISK_ROOT/venv"
fi
source "$DISK_ROOT/venv/bin/activate"

echo "==> Installing vLLM (this pulls a matching torch, several GB)"
pip install -q --upgrade pip
pip install -q vllm

# Local speech-to-text, so microphone audio never leaves the machine. CPU only:
# the GPUs are taken by the vision model.
echo "==> Installing faster-whisper and the STT service deps"
pip install -q faster-whisper fastapi uvicorn python-multipart piper-tts

echo "==> vLLM version:"
python -c "import vllm; print(vllm.__version__)"

echo "==> Pre-downloading $MODEL into $HF_HOME (~20GB, slow)"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL")
print("done")
PY

# Both sizes, because the job runs with HF_HUB_OFFLINE=1: asking for a size that
# was never downloaded kills the speech service at startup, and the browser only
# sees "failed to fetch".
echo "==> Pre-downloading the Whisper weights (base + small, ~650MB)"
python - <<WHISPER
from faster_whisper import WhisperModel
for size in ("base", "small"):
    print(f"  {size}...", flush=True)
    WhisperModel(size, device="cpu", compute_type="int8")
print("done")
WHISPER

# Piper voice for the answers. Local neural speech: the system voices on a laptop
# sound robotic, and the good-sounding browser voices synthesise in the cloud.
echo "==> Downloading the Piper voice (~60MB)"
mkdir -p "$DISK_ROOT/piper"
python - <<VOICE
from huggingface_hub import hf_hub_download
import shutil, pathlib

dest = pathlib.Path("$DISK_ROOT/piper")
base = "es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
for name in (base, base + ".json"):
    src = hf_hub_download("rhasspy/piper-voices", name)
    shutil.copy(src, dest / pathlib.Path(name).name)
    print(" ", pathlib.Path(name).name)
VOICE

echo
echo "Setup complete. Now submit the server job:"
echo "  sbatch cluster/vllm.sbatch"
