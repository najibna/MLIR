"""export a small pytorch mlp into our mlir-ish text form"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    # two layer net we use for demos
    def __init__(self, in_dim=4, hidden=8, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


def tensor_to_list(t: torch.Tensor):
    return [float(v) for v in t.detach().cpu().reshape(-1).tolist()]


def export_mlir(model: TinyMLP, path: Path, batch: int = 1):
    # write nodes in the same style the c++ side can parse
    w1 = model.fc1.weight.detach().cpu().t().contiguous()
    w2 = model.fc2.weight.detach().cpu().t().contiguous()
    in_dim = w1.shape[0]
    hidden = w1.shape[1]
    out_dim = w2.shape[1]

    lines = [
        "module {",
        "  func @main() {",
        f"    %0 = nn.input : tensor<{batch}x{in_dim}xf32>  // x",
        f"    %1 = nn.constant : tensor<{in_dim}x{hidden}xf32>  // w1",
        f"    %2 = nn.matmul (%0, %1) : tensor<{batch}x{hidden}xf32>  // fc1",
        f"    %3 = nn.relu (%2) : tensor<{batch}x{hidden}xf32>  // relu1",
        f"    %4 = nn.constant : tensor<{hidden}x{out_dim}xf32>  // w2",
        f"    %5 = nn.matmul (%3, %4) : tensor<{batch}x{out_dim}xf32>  // fc2",
        f"    %6 = nn.softmax (%5) : tensor<{batch}x{out_dim}xf32>  // probs",
        f"    %7 = nn.output (%6) : tensor<{batch}x{out_dim}xf32>  // y",
        "    return",
        "  }",
        "}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))

    # also dump weights next to it so the runtime helpers can load them
    weight_path = path.with_suffix(".weights.txt")
    with weight_path.open("w") as f:
        f.write("w1 " + " ".join(str(v) for v in tensor_to_list(w1)) + "\n")
        f.write("w2 " + " ".join(str(v) for v in tensor_to_list(w2)) + "\n")
    return path, weight_path


def main():
    p = argparse.ArgumentParser(description="export pytorch mlp to mlir text")
    p.add_argument("--out", default="build/model.mlir")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = TinyMLP()
    model.eval()
    mlir_path, w_path = export_mlir(model, Path(args.out))
    print(f"wrote {mlir_path}")
    print(f"wrote {w_path}")

    # quick sanity check against pytorch
    x = torch.tensor([[0.5, -0.25, 0.75, 0.1]])
    with torch.no_grad():
        y = model(x)
    print("pytorch output:", y.tolist())


if __name__ == "__main__":
    main()
