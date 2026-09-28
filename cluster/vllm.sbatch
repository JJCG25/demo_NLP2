#!/bin/bash
#SBATCH --job-name=qwen-vl
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=vllm-%j.out
#SBATCH --error=vllm-%j.err
#
# Serves Qwen2.5-VL-32B-AWQ across the 4 T4s of one node.
#
#   sbatch cluster/vllm.sbatch
#   tail -f vllm-<jobid>.out      # prints the exact SSH tunnel command
#
# Raised from the template you were given: --gres=gpu:4 (we need all four),
# --cpus-per-task=16 and --mem=64G (TP=4 runs one worker process per GPU and
# stages weights through host RAM; 16G gets the job OOM-killed).
#
# Slurm does NOT expand $USER or $HOME in #SBATCH lines, so the log paths above
# are relative to wherever you run sbatch from: clone the repo under /disk and
# submit from there, or hardcode an absolute path in the two directives.

set -euo pipefail

# --- EDIT THIS if your space is somewhere else ---
DISK_ROOT="/disk/$USER"
# -------------------------------------------------

MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

source "$DISK_ROOT/venv/bin/activate"

# Keep every cache off $HOME: weights, torch.compile and triton artifacts all
# grow quietly and home quotas are usually small.
export HF_HOME="$DISK_ROOT/hf"
export VLLM_CACHE_ROOT="$DISK_ROOT/vllm-cache"
export TRITON_CACHE_DIR="$DISK_ROOT/vllm-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$DISK_ROOT/vllm-cache/inductor"
# Catch-all for libraries that ignore the specific vars above and just use
# ~/.cache, plus the NVIDIA JIT cache, which defaults to ~/.nv.
export XDG_CACHE_HOME="$DISK_ROOT/cache"
export CUDA_CACHE_PATH="$DISK_ROOT/cache/nv"
mkdir -p "$DISK_ROOT"/cache/nv

# Weights were pre-downloaded by setup_env.sh. Staying offline means the job
# fails fast with a clear error instead of hanging on a blocked network if the
# compute nodes have no internet access.
export HF_HUB_OFFLINE=1

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Derived from the job id so two jobs on one node cannot collide. The browser
# side of the tunnel is always 8000 and 8100, so script.js needs no edit.
PORT=$((8000 + SLURM_JOB_ID % 1000))
PORT_STT=$((9000 + SLURM_JOB_ID % 1000))

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES | CPUs: $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo
echo "=============================================================="
echo " Once you see 'Application startup complete', run THIS on your"
echo " laptop (replace the login host), and keep it open:"
echo
echo "   ssh -L 8000:$SLURMD_NODENAME:$PORT \\"
echo "       -L 8100:$SLURMD_NODENAME:$PORT_STT <user>@<login-host>"
echo
echo " Then serve the page locally: python serve.py 8080"
echo "=============================================================="
echo

# Whisper for speech, on the CPUs that sit idle while the GPUs run the vision
# model, so microphone audio never leaves the machine. Backgrounded: the job
# lives and dies with vLLM below, and this goes with it.
#
# STT_MODEL=base is roughly twice as fast and noticeably less accurate;
# STT_MODEL=medium is the other way round. Override when submitting:
#   STT_MODEL=base sbatch cluster/vllm.sbatch
STT_THREADS=12 STT_MODEL="${STT_MODEL:-small}" python -m uvicorn cluster.stt_server:app \
    --host 0.0.0.0 --port "$PORT_STT" --log-level warning &
STT_PID=$!
trap 'kill $STT_PID 2>/dev/null' EXIT
echo "STT (whisper, cpu) starting on port $PORT_STT, pid $STT_PID"
echo

# T4 is Turing (sm75): float16 is mandatory, since the checkpoint is bf16 and
# sm75 has no bfloat16 support. FlashAttention-2 is Ampere-only, so attention
# falls back to a slower kernel -- expected, not a misconfiguration.
#
# --host 0.0.0.0 is what lets the login node reach this port for the tunnel.
# --served-model-name matters because of HF_HUB_OFFLINE: vLLM resolves the repo
# id to the local snapshot path and would otherwise register the model under
# that path, so requests naming the repo id get a 404. This pins the public name
# to what script.js and bench.py send.
srun vllm serve "$MODEL" \
    --served-model-name "$MODEL" \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-kwargs '{"max_pixels": 589824}' \
    --host 0.0.0.0 \
    --port "$PORT"

# Tuning notes:
#
# --tensor-parallel-size 4   int4 weights are ~20GB so 2 cards would fit, but
#                            splitting across 4 leaves more room for KV cache
#                            and puts 4 GPUs of compute on each token. T4s have
#                            no NVLink, so this costs PCIe traffic -- worth
#                            timing TP=2 against TP=4 on your node.
# --max_pixels 589824        768x768, matching MAX_SIDE in script.js. Main
#                            latency dial: fewer pixels, faster prefill.
#
# If it OOMs on the GPUs: lower --gpu-memory-utilization to 0.85, then add
# --enforce-eager, then lower --max-model-len.
# If Slurm kills the job for memory: raise --mem above.
