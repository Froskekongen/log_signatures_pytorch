"""Microbenchmark comparing Hall vs words projection paths.

Run:
    PYTHONPATH=src uv run python benchmarks/words_vs_hall.py --width 3 --depth 4
"""

from __future__ import annotations

import argparse
import time

import torch

from log_signatures_pytorch import log_signature


def benchmark(width: int, depth: int, batch: int, length: int, device: str):
    torch.manual_seed(0)
    path = torch.randn(batch, length, width, device=device)

    def run(mode: str):
        torch.cuda.empty_cache() if device.startswith("cuda") else None
        torch.cuda.synchronize() if device.startswith("cuda") else None
        start = time.perf_counter()
        out = log_signature(path, depth=depth, mode=mode)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        elapsed = (time.perf_counter() - start) * 1e3
        return elapsed, float(out.norm().cpu())

    hall_ms, hall_norm = run("hall")
    words_ms, words_norm = run("words")

    print(f"device={device} width={width} depth={depth} batch={batch} length={length}")
    print(f"  hall : {hall_ms:7.2f} ms  | norm={hall_norm:.4f}")
    print(f"  words: {words_ms:7.2f} ms  | norm={words_norm:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    benchmark(
        width=args.width,
        depth=args.depth,
        batch=args.batch,
        length=args.length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
