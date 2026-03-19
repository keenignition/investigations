#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
mkdir -p out

# Default N for profiling (use 4096 so fused uses register kernel; 8192 would use block kernel)
N="${1:-4096}"

run_ncu() {
  local name="$1"
  local provider="$2"
  echo "[ncu] $name (provider=$provider, N=$N)"
  ncu --set full -f -o "out/${name}" python scripts/run_kernel.py "$provider" "$N"
}

run_ncu "softmax_naive"   "softmax_naive"
run_ncu "softmax_wr"      "softmax_wr"
run_ncu "softmax_fused"   "softmax_fused"
run_ncu "fused_triton"    "fused"

echo "Reports written to $ROOT/out/*.ncu-rep"
