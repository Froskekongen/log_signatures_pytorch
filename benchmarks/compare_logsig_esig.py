#!/usr/bin/env python3
"""Benchmark log_signatures_pytorch against esig.

This script measures runtime and numerical agreement between
`log_signatures_pytorch.log_signature` (batched) and `esig.stream2logsig`
across configurable widths, depths, and path lengths.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from typing import Iterable, List, Sequence

import torch

from benchmarks.utils import generate_paths, maybe_sync, time_call_once

try:
    import esig
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "The esig package is required for this benchmark. "
        "Install the optional dependency group first."
    ) from exc

from log_signatures_pytorch.log_signature import log_signature


def _format_seconds(value: float) -> str:
    return f"{value * 1e3:8.2f} ms"


def benchmark(
    lengths: Sequence[int],
    widths: Sequence[int],
    depths: Sequence[int],
    batch_size: int,
    repeats: int,
    warmup: int,
    atol: float,
    rtol: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    mode: str,
) -> List[dict]:
    rows: List[str] = []
    header = "width depth length batch esig_time logsig_time speedup max_err"
    rows.append(header)
    records: List[dict] = []
    for width, depth, length in itertools.product(widths, depths, lengths):
        log_times: List[float] = []
        esig_times: List[float] = []
        max_errors: List[float] = []
        warmup_paths_cpu = generate_paths(
            batch=batch_size,
            length=length,
            width=width,
            dtype=dtype,
            seed=seed,
        )
        if device.type == "cuda":
            warmup_paths_cpu = warmup_paths_cpu.pin_memory()
        warmup_paths = (
            warmup_paths_cpu.to(device, non_blocking=True)
            if device.type == "cuda"
            else warmup_paths_cpu
        )

        for _ in range(warmup):
            log_signature(
                warmup_paths,
                depth=depth,
                stream=False,
                mode=mode,
            )
        maybe_sync(device)

        def _esig_eval(paths_cpu: torch.Tensor) -> torch.Tensor:
            vals = [
                torch.tensor(
                    esig.stream2logsig(path.numpy(), depth),
                    dtype=dtype,
                )
                for path in paths_cpu
            ]
            return torch.stack(vals, dim=0)

        for _ in range(warmup):
            _esig_eval(warmup_paths_cpu)

        for repeat in range(repeats):
            paths_cpu = generate_paths(
                batch=batch_size,
                length=length,
                width=width,
                dtype=dtype,
                seed=seed + repeat,
            )
            if device.type == "cuda":
                paths_cpu = paths_cpu.pin_memory()
            paths = (
                paths_cpu.to(device, non_blocking=True)
                if device.type == "cuda"
                else paths_cpu
            )
            ours, t_log = time_call_once(
                lambda p: log_signature(
                    p,
                    depth=depth,
                    stream=False,
                    mode=mode,
                ),
                paths,
                device=device,
            )
            log_times.append(t_log)

            ours_cpu = ours.cpu()
            ref, t_esig = time_call_once(
                _esig_eval,
                paths_cpu,
                device=torch.device("cpu"),
            )
            esig_times.append(t_esig)
            if ours_cpu.shape != ref.shape:
                raise RuntimeError(
                    f"Shape mismatch: ours {ours_cpu.shape} vs esig {ref.shape}"
                )
            torch.testing.assert_close(
                ours_cpu,
                ref,
                atol=atol,
                rtol=rtol,
                msg=f"Outputs differ for width={width}, depth={depth}, length={length}",
            )
            max_errors.append((ours_cpu - ref).abs().max().item())

        mean_log = statistics.fmean(log_times)
        mean_esig = statistics.fmean(esig_times)
        speedup = mean_esig / mean_log if mean_log > 0 else float("inf")
        records.append(
            {
                "width": width,
                "depth": depth,
                "length": length,
                "batch": batch_size,
                "esig_time_ms": mean_esig * 1e3,
                "logsig_time_ms": mean_log * 1e3,
                "speedup": speedup,
                "max_err": max(max_errors),
            }
        )
        rows.append(
            f"{width:>5} {depth:>5} {length:>6} {batch_size:>5} "
            f"{_format_seconds(mean_esig)} {_format_seconds(mean_log)} "
            f"{speedup:7.2f}x {max(max_errors):.2e}"
        )
    print("\n".join(rows))
    return records


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark log_signatures_pytorch against esig."
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[2, 3],
        help="List of path widths to benchmark.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="List of log-signature depths.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="List of path lengths (timesteps).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of paths evaluated per batch.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per configuration.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for equality checks.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for equality checks.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Floating point precision to use.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for log_signature timing (esig always runs on CPU).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations before timing each function.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write CSV results for regression tracking.",
    )
    parser.add_argument(
        "--require-speedup",
        type=float,
        default=None,
        help="If set, fail when logsig_time > esig_time * factor (e.g., 1.5).",
    )
    parser.add_argument(
        "--mode",
        choices=["hall"],
        default="hall",
        help=(
            "Basis to use for log_signature; esig outputs Hall coordinates, so"
            " comparison is only valid for mode='hall'."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    torch.set_grad_enabled(False)
    mode = args.mode.lower()
    if mode != "hall":  # pragma: no cover - defensive guard
        raise SystemExit("Comparison with esig is only supported in Hall basis.")

    records = benchmark(
        lengths=args.lengths,
        widths=args.widths,
        depths=args.depths,
        batch_size=args.batch_size,
        repeats=args.repeats,
        warmup=args.warmup,
        atol=args.atol,
        rtol=args.rtol,
        dtype=dtype,
        device=device,
        seed=args.seed,
        mode=mode,
    )
    if args.output_csv:
        import csv

        fieldnames = [
            "width",
            "depth",
            "length",
            "batch",
            "esig_time_ms",
            "logsig_time_ms",
            "speedup",
            "max_err",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    if args.require_speedup is not None:
        factor = args.require_speedup
        slow_cases = [
            r for r in records if r["logsig_time_ms"] > r["esig_time_ms"] * factor
        ]
        if slow_cases:
            details = ", ".join(
                f"(w={r['width']},d={r['depth']},L={r['length']},speedup={r['speedup']:.2f}x)"
                for r in slow_cases
            )
            raise SystemExit(
                f"Speedup requirement not met (factor={factor}). Slow cases: {details}"
            )


if __name__ == "__main__":
    main()
