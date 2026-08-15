# smoke checks for the python helpers

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from compile import fuse_text  # noqa: E402


def test_fuse_matmul_relu():
    src = """module {
  func @main() {
    %0 = nn.input : tensor<1x4xf32>  // x
    %1 = nn.constant : tensor<4x8xf32>  // w1
    %2 = nn.matmul (%0, %1) : tensor<1x8xf32>  // fc1
    %3 = nn.relu (%2) : tensor<1x8xf32>  // relu1
    %4 = nn.output (%3) : tensor<1x8xf32>  // y
    return
  }
}
"""
    out = fuse_text(src)
    assert "fused_matmul_relu" in out
    assert "nn.relu" not in out
    assert re.search(r"output \(%2\)", out)


if __name__ == "__main__":
    test_fuse_matmul_relu()
    print("ok")
