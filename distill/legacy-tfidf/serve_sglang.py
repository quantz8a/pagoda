"""Optional: deploy a small *autoregressive* model locally with SGLang.

Contrast with serve.py:
  - serve.py          -> the distilled DECISION head. Non-autoregressive: one
                         forward over the text, no prefill/decode. No SGLang.
  - serve_sglang.py   -> a small GENERATIVE LLM. Autoregressive: it *does* use
                         prefill/decode, which is exactly what SGLang serves
                         well (radix prefix cache + continuous batching).

Requires: `pip install sglang`, network to fetch weights, and ideally a GPU.
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="any HF instruct model id, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=30000)
    args = ap.parse_args()

    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model-path", args.model,
           "--host", args.host, "--port", str(args.port)]
    print("launching SGLang (prefill/decode serving):", " ".join(cmd))
    print("OpenAI-compatible endpoint: http://%s:%d/v1" % (args.host, args.port))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()