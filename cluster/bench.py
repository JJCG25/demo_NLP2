#!/usr/bin/env python3
"""Measures vLLM latency the way the app actually uses it: one image, one short
question, one answer.

Run it on the compute node against localhost so no network is in the numbers:

    python cluster/bench.py --port 8123
    python cluster/bench.py --port 8123 --sweep 512,768,1024

Reports time to first token (how long prefill + vision encoding take, which is
where the image size shows up) and decode speed (tokens/s, where model size and
tensor parallelism show up). Those two add up to what the user waits for.
"""
import argparse
import base64
import io
import json
import statistics
import time

import requests
from PIL import Image, ImageDraw


def make_frame(side: int) -> str:
    """A synthetic 4:3 frame with some structure, so the vision encoder has real
    work to do rather than compressing a flat color."""
    w, h = side, int(side * 3 / 4)
    img = Image.new("RGB", (w, h), (30, 40, 60))
    draw = ImageDraw.Draw(img)
    for i in range(0, w, 40):
        draw.line([(i, 0), (i + 120, h)], fill=(90, 110, 140), width=3)
    for i in range(6):
        x, y = 40 + i * (w // 7), h // 3
        draw.ellipse([x, y, x + 60, y + 60], fill=(200, 160 - i * 20, 80))
    draw.text((20, h - 40), "BENCHMARK FRAME", fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def one_request(url, model, b64, prompt, max_tokens):
    """Returns (ttft, total, completion_tokens). Streams so first-token time is
    measurable separately from the full answer."""
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Eres un asistente experto en análisis visual. "
                           "Responde SIEMPRE en español, de forma concisa y directa.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    started = time.perf_counter()
    ttft = None
    completion_tokens = 0
    text = []

    with requests.post(url, json=body, stream=True, timeout=600) as res:
        res.raise_for_status()
        for line in res.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)

            if chunk.get("usage"):
                completion_tokens = chunk["usage"]["completion_tokens"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    text.append(piece)

    total = time.perf_counter() - started
    return ttft or total, total, completion_tokens, "".join(text).strip()


def bench_one_size(url, model, side, prompt, max_tokens, runs):
    b64 = make_frame(side)
    kb = len(b64) * 3 // 4 // 1024

    # Warmup: the first request pays for lazy init and cache warming, and would
    # otherwise skew a small sample badly.
    _, _, _, sample = one_request(url, model, b64, prompt, max_tokens)

    ttfts, totals, toks = [], [], []
    for _ in range(runs):
        ttft, total, n, _ = one_request(url, model, b64, prompt, max_tokens)
        ttfts.append(ttft)
        totals.append(total)
        toks.append(n)

    med_total = statistics.median(totals)
    med_ttft = statistics.median(ttfts)
    med_tok = statistics.median(toks)
    decode_rate = (med_tok - 1) / (med_total - med_ttft) if med_total > med_ttft else 0

    print(f"\n{side}px frame ({kb} KB JPEG), {runs} runs")
    print(f"  time to first token : {med_ttft:5.2f}s   (prefill + vision)")
    print(f"  total               : {med_total:5.2f}s   "
          f"[min {min(totals):.2f} / max {max(totals):.2f}]")
    print(f"  decode              : {decode_rate:5.1f} tok/s over {med_tok:.0f} tokens")
    print(f"  answer              : {sample[:100]}")
    return med_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True,
                    help="the port from the job log (8000 + jobid %% 1000)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
    ap.add_argument("--prompt", default="¿Qué ves en la imagen?")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sweep", default="768",
                    help="comma-separated frame sizes, e.g. 512,768,1024")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print(f"target : {url}")
    print(f"model  : {args.model}")
    print(f"budget : {args.max_tokens} max tokens")

    results = {}
    for side in [int(s) for s in args.sweep.split(",")]:
        results[side] = bench_one_size(
            url, args.model, side, args.prompt, args.max_tokens, args.runs
        )

    if len(results) > 1:
        print("\nsummary (median total):")
        for side, total in results.items():
            print(f"  {side:>5}px  {total:5.2f}s")
        print("\nPick the smallest frame whose answers are still good enough, "
              "then set MAX_SIDE in script.js and max_pixels in the sbatch to match.")


if __name__ == "__main__":
    main()
