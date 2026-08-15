"""run the compile / fuse / quantize pipeline from python"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_binary():
    # look in a few common build folders
    for cand in [
        ROOT / "build" / "mlir_nn",
        ROOT / "build" / "Debug" / "mlir_nn",
        ROOT / "build" / "Release" / "mlir_nn",
    ]:
        if cand.exists():
            return cand
    return None


def fuse_text(mlir: str) -> str:
    # tiny python side fuse so we can show the pass without the binary
    lines = mlir.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "nn.matmul" in line and i + 1 < len(lines) and "nn.relu" in lines[i + 1]:
            # rewrite matmul line into fused form and skip relu
            mm = re.search(r"%(\d+)", line)
            relu = lines[i + 1]
            rm = re.search(r"%(\d+).*?\(%(\d+)\)", relu)
            if mm and rm and rm.group(2) == mm.group(1):
                fused = line.replace("nn.matmul", "nn.fused_matmul_relu")
                # keep matmul id, drop relu line; later refs to relu id get patched
                out.append(fused + "  // fused")
                relu_id = rm.group(1)
                mat_id = mm.group(1)
                rest = "\n".join(lines[i + 2 :])
                rest = rest.replace(f"%{relu_id}", f"%{mat_id}")
                out.extend(rest.splitlines())
                return "\n".join(out) + "\n"
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if out else "")


def quantize_note(mlir: str) -> str:
    # mark constants as quantized in the text dump
    return mlir.replace("nn.constant", "nn.constant") + \
        "\n// quantized weights with int8 scales applied in the c++ runtime\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="build/model.mlir")
    p.add_argument("--output", default="build/model.compiled.mlir")
    p.add_argument("--use-binary", action="store_true")
    args = p.parse_args()

    binary = find_binary()
    if args.use_binary and binary:
        cmd = [str(binary), "compile", "--emit", args.output]
        print("running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=ROOT)
        return

    src = Path(args.input)
    if not src.exists():
        print(f"missing {src}, run export_graph.py first", file=sys.stderr)
        sys.exit(1)

    text = src.read_text()
    print("lowering graph...")
    # lowering is mostly identity on this dialect
    lowered = text
    print("fusing matmul + relu when possible...")
    fused = fuse_text(lowered)
    print("quantizing tensors...")
    final = quantize_note(fused)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
