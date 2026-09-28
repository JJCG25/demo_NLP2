# Qwen2.5-VL Vision Assistant Demo

A web application that answers questions about what it sees through your camera,
in Spanish, using **Qwen2.5-VL-32B-Instruct-AWQ** served by vLLM on 4x Tesla T4.

The browser holds the camera and the UI. The model runs on the GPU box and is
reached over an SSH tunnel, so the page only ever talks to `localhost`.

```
laptop            login node         compute node (Slurm job)
------            ----------         -----------------------
camera + UI  -->  ssh hop      -->   vLLM, 4x T4, TP=4
   :8000                             Qwen2.5-VL-32B-AWQ
         \_______ ssh -L tunnel _______/
```

## ✨ Features
- **Immersive UI**: Full-screen camera feed with a modern glassmorphism control panel.
- **Draggable Controls**: Move the interaction panel anywhere on your screen.
- **Spanish answers**: The system prompt pins the response language.
- **Timing readout**: Each answer reports its latency, and the console logs
  tokens and frame size so you can tune.

## 🚀 Getting Started

### 1. Prerequisites
- **Cluster access**: a Slurm cluster with a 4x T4 node, and space on `/disk`
  (~25 GB for the weights and the venv).
- **Laptop**: any modern browser and a webcam. No WebGPU needed any more.

### 2. One-time setup (on the login node)
Clone the repo **under `/disk`**, not in `$HOME` — Slurm writes job logs into the
directory you submit from.
```bash
cd /disk/$USER && git clone <your-repo-url> && cd demo_NLP2
bash cluster/setup_env.sh
```
Creates the venv, installs vLLM and pre-downloads the ~20 GB of weights, all
under `/disk/$USER`. The download happens here because compute nodes often have
no internet access. Edit `DISK_ROOT` at the top if your space is elsewhere.

### 3. Start the server (from the login node)
```bash
sbatch cluster/vllm.sbatch
tail -f vllm-<jobid>.out
```
The job takes all 4 GPUs. Its log prints the exact tunnel command, with the
compute node and port already filled in — copy it from there.

Note the **12-hour time limit** in the script: the server dies when the job
ends, and you resubmit. Raise it to cover a whole event (the `main` partition
has no limit of its own), but do not park a job on all four GPUs longer than you
need it — the machine is shared. The weights are cached on `/disk`, so later
starts only pay the load time, not the download.

### 4. Open the tunnel (on the laptop)
```bash
ssh -L 8000:<node>:<vllm-port> -L 8100:<node>:<stt-port> user@login-host
```
Both ports come from the job log. Leave it running. Two hops in one command:
your laptop reaches the login node, which reaches the compute node. This is why
no CORS setup or exposed port is needed.

Port 8000 carries the vision model, 8100 the speech-to-text service.

### 5. Serve the page (on the laptop)
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

## 📏 Measuring latency

Measure on the node against `localhost`, so neither the network nor the browser
is in the numbers. The benchmark only makes HTTP calls, so it needs no GPU and
no Slurm allocation — just a second shell on the same machine while the server
runs:

```bash
source /disk/$USER/venv/bin/activate
python cluster/bench.py --port <port>                        # single size
python cluster/bench.py --port <port> --sweep 512,768,1024   # find the knee
```

Under `HF_HUB_OFFLINE` a server started by hand registers the model under its
local snapshot path, so pass the name the server reports:
```bash
--model "$(curl -s localhost:<port>/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
```
The sbatch pins the name with `--served-model-name`, so this is only needed for
a hand-started server.

It reports two numbers per frame size, and they have different causes:

- **Time to first token** — prefill plus vision encoding. This is what grows
  with image size, so it's the number `MAX_SIDE` moves.
- **Decode tok/s** — steady-state generation. This is what model size and
  tensor parallelism move; `MAX_TOKENS` multiplies it.

Add them and you have what the user waits for. If TTFT dominates, shrink the
frame. If decode dominates, shorten the answer or try TP=2 against TP=4.

The browser also prints its own latency under each answer, and logs tokens and
frame size to the console — that one includes the tunnel, so comparing the two
tells you what the network is costing you (should be milliseconds).

## ⚡ Tuning latency
Expect a few seconds per answer. T4s are pre-Ampere, so there is no
FlashAttention-2 and no bf16, and a 32B model is genuinely heavy. In order of
impact:

| Knob | Where | Effect |
|---|---|---|
| `MAX_SIDE` | [script.js](script.js) | Frame resolution. Fewer pixels, fewer vision tokens, faster prefill. Keep `max_pixels` in the sbatch in sync. |
| `MAX_TOKENS` | [script.js](script.js) | Caps answer length. Decoding is token-by-token, so this scales latency directly. |
| `--tensor-parallel-size` | [cluster/vllm.sbatch](cluster/vllm.sbatch) | 4 spreads compute but costs PCIe traffic (no NVLink on T4). Measure 2 against 4. |

If it's still too slow, `Qwen/Qwen2.5-VL-7B-Instruct` in fp16 fits on a single
T4 and is several times faster. Change `MODEL_ID` in [script.js](script.js), and
`MODEL` plus `--gres=gpu:1` and `--tensor-parallel-size 1` in the sbatch.

To experiment without queueing a batch job each time, grab an interactive
allocation and run the same `vllm serve` command by hand:
```bash
salloc --partition=main --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=1:00:00
```

## 📝 Technical Notes
- **Model**: [`Qwen/Qwen2.5-VL-32B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ) (int4, ~20 GB)
- **Engine**: vLLM, OpenAI-compatible `/v1/chat/completions`, inside a Slurm job
- **Speech in**: Whisper `small` (int8) on the CPUs via faster-whisper, in the
  same job. Audio never leaves the machine; set `STT_BACKEND = "browser"` in
  `script.js` to use Chrome's recogniser instead, which streams it to Google
- **Speech out**: the browser's `speechSynthesis`, preferring a voice with
  `localService` so the answer text is synthesised on the laptop. Chrome's
  "Google ..." voices render on Google's servers; the console logs which one was
  picked, so check it before a public demo
- **Storage**: venv, HF cache and compile caches all under `/disk/$USER`, never `$HOME`
- **Precision**: `float16`, required — T4 (sm75) has no bfloat16 support
- **Why not Qwen3-VL**: it has [no vLLM backend for Turing GPUs](https://github.com/vllm-project/vllm/issues/29743)
- **`serve.py`**: the COOP/COEP headers it sends are harmless leftovers from the
  in-browser era; `python -m http.server 8080` now works just as well.
