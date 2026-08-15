"""benchmark inference latency for the compiled graph"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 3, bias=False)

    def forward(self, x):
        return torch.softmax(self.fc2(torch.relu(self.fc1(x))), dim=-1)


def bench_pytorch(iters: int, warmup: int):
    model = TinyMLP().eval()
    x = torch.randn(1, 4)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_binary(iters: int, use_cuda: bool):
    binary = ROOT / "build" / "mlir_nn"
    if not binary.exists():
        return None
    cmd = [str(binary), "bench", "--iters", str(iters)]
    cmd.append("--cuda" if use_cuda else "--cpu")
    out = subprocess.check_output(cmd, text=True, cwd=ROOT)
    latency = None
    for line in out.splitlines():
        if line.startswith("latency_ms:"):
            latency = float(line.split(":")[1].strip())
    return latency, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    print("pytorch reference bench")
    times = bench_pytorch(args.iters, args.warmup)
    print(f"  mean_ms: {statistics.mean(times):.4f}")
    print(f"  p50_ms:  {statistics.median(times):.4f}")
    print(f"  p95_ms:  {sorted(times)[int(0.95 * (len(times) - 1))]:.4f}")

    result = bench_binary(args.iters, args.cuda)
    if result is None:
        print("mlir_nn binary not found, skip native bench (build first)")
        return

    latency, raw = result
    print("native runtime bench")
    print(raw.strip())
    if latency is not None:
        print(f"  parsed_latency_ms: {latency:.4f}")


if __name__ == "__main__":
    main()
