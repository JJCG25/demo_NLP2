#!/usr/bin/env bash
# Runs on the 4x T4 box, not on the laptop. vLLM has no native Windows build.
#
#   pip install vllm
#   ./run_vllm.sh
#
# First start downloads ~20GB of int4 weights and takes a while. Then, from the
# laptop, open the tunnel so the browser can reach it as localhost:
#
#   ssh -L 8000:localhost:8000 user@t4-box
set -euo pipefail

# T4 is Turing (sm75). float16 is mandatory: bfloat16 needs compute capability
# 8.0+, and the Qwen checkpoints ship in bf16, so without this vLLM refuses to
# start. FlashAttention-2 is also Ampere-only, so attention falls back to a
# slower kernel -- expected, not a misconfiguration.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# If startup dies on an attention backend error, uncomment:
# export VLLM_ATTENTION_BACKEND=XFORMERS

vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-kwargs '{"max_pixels": 589824}' \
    --port 8000

# Notes on the knobs above:
#
# --tensor-parallel-size 4   int4 weights are ~20GB, so 2 cards would fit, but
#                            splitting across all 4 leaves far more room for KV
#                            cache and puts 4 GPUs of compute on each token.
#                            T4s have no NVLink, so this costs PCIe traffic --
#                            worth measuring TP=2 against TP=4 on your box.
# --max-model-len 4096       Plenty for one frame plus a short answer, and keeps
#                            the KV cache small. Raise it only if you need it.
# --max_pixels 589824        768x768, matching MAX_SIDE in script.js. This is
#                            the main latency dial: fewer pixels, fewer vision
#                            tokens, faster prefill.
#
# If you hit OOM at startup, in order: lower --gpu-memory-utilization to 0.85,
# add --enforce-eager (frees CUDA graph memory), then lower --max-model-len.
