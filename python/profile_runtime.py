"""profile and schedule helpers around the cuda / cpu runtime"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_profile(use_cuda: bool) -> str:
    binary = ROOT / "build" / "mlir_nn"
    if not binary.exists():
        return (
            "mlir_nn binary missing\n"
            "build with: cmake -S . -B build && cmake --build build\n"
        )
    cmd = [str(binary), "profile", "--cuda" if use_cuda else "--cpu"]
    return subprocess.check_output(cmd, text=True, cwd=ROOT)


def estimate_schedule(ops):
    # same rough costs as the c++ scheduler
    costs = {
        "matmul": 0.002,
        "fused_matmul_relu": 0.002,
        "softmax": 0.001,
        "relu": 0.0002,
        "add": 0.0002,
    }
    order = []
    total = 0.0
    for name, elems in ops:
        order.append(name)
        total += costs.get(name, 0.0001) * elems
    return order, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--python-only", action="store_true")
    args = p.parse_args()

    if args.python_only:
        ops = [
            ("input", 4),
            ("matmul", 32),
            ("relu", 8),
            ("matmul", 24),
            ("softmax", 3),
            ("output", 3),
        ]
        order, est = estimate_schedule(ops)
        print("schedule:", " -> ".join(order))
        print(f"estimated_ms: {est:.6f}")
        return

    print(run_profile(args.cuda))


if __name__ == "__main__":
    main()
