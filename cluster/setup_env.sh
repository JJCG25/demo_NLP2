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
mkdir -p "$DISK_ROOT"/{hf,logs,vllm-cache}

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

echo "==> vLLM version:"
python -c "import vllm; print(vllm.__version__)"

echo "==> Pre-downloading $MODEL into $HF_HOME (~20GB, slow)"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL")
print("done")
PY

echo
echo "Setup complete. Now submit the server job:"
echo "  sbatch cluster/vllm.sbatch"
