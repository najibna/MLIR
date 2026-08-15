"""end to end demo: export, compile, and print what each stage did"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "python"


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    run([sys.executable, str(PY / "export_graph.py"), "--out", "build/model.mlir"])
    run([sys.executable, str(PY / "compile.py"), "--input", "build/model.mlir",
         "--output", "build/model.compiled.mlir"])
    run([sys.executable, str(PY / "profile_runtime.py"), "--python-only"])
    run([sys.executable, str(PY / "benchmark.py"), "--iters", "30"])

    binary = ROOT / "build" / "mlir_nn"
    if binary.exists():
        run([str(binary), "compile", "--emit", "build/demo.mlir"])
        run([str(binary), "run", "--cpu"])
        run([str(binary), "bench", "--cpu", "--iters", "30"])
        run([str(binary), "profile", "--cpu"])
    else:
        print("\nnote: build/mlir_nn not found yet, python stages still ran")


if __name__ == "__main__":
    main()
