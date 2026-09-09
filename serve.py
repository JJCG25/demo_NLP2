"""Static server with the cross-origin isolation headers that transformers.js
needs for multi-threaded WASM (SharedArrayBuffer).

Usage:  python serve.py [port]     (default port 8000)

`python -m http.server` does NOT send these headers, which silently forces
ONNX Runtime to run single-threaded and makes inference much slower.
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving http://localhost:{port}/  (COOP/COEP enabled)")
    ThreadingHTTPServer(("localhost", port), Handler).serve_forever()
