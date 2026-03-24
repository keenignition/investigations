#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
mkdir -p out

# Default N for profiling (use 4096 so fused uses register kernel; 8192 would use block kernel)
N="${1:-4096}"

run_nsys() {
  local name="$1"
  local provider="$2"
  echo "[nsys] $name (provider=$provider, N=$N)"
  nsys profile -t cuda,nvtx -f true -o "out/${name}" python run_kernel.py "$provider" "$N"
}

run_nsys "softmax_naive"         "softmax_naive"
run_nsys "softmax_wr"            "softmax_wr"
run_nsys "fused_triton"          "fused"
run_nsys "softmax_fused_warp"    "softmax_fused_warp"
run_nsys "softmax_fused_block"   "softmax_fused_block"

echo "Reports written to $ROOT/out/*.nsys-rep"
