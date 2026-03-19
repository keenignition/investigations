#!/usr/bin/env bash
# Dump SASS and PTX for a kernel from an NCU report to text files (for grepping, diffing, or scripts).
#
# Usage:
#   ./ncu_dump_sass.sh <report.ncu-rep> [kernel_regex] [output_dir]
#
# Examples:
#   ./ncu_dump_sass.sh out/softmax_naive.ncu-rep
#   ./ncu_dump_sass.sh out/softmax_naive.ncu-rep "softmax_kernel" out/dumps
#
# Output: <output_dir>/<report_stem>_<kernel_slug>.sass and .ptx
# If kernel_regex is omitted, the first non–at:: kernel in the report is used (name from ncu output).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

REP="${1:?Usage: $0 <report.ncu-rep> [kernel_regex] [output_dir]}"
KERNEL_REGEX="${2:-}"
OUT_DIR="${3:-$ROOT/out/ncu_dumps}"

REP_PATH="$(realpath -e "$REP")"
REP_NAME="$(basename "$REP_PATH" .ncu-rep)"
mkdir -p "$OUT_DIR"

if [ -n "$KERNEL_REGEX" ]; then
  KFILTER="-k"
  KARG="$KERNEL_REGEX"
  # Slug for filename: first word of regex, lowercased
  KERNEL_SLUG="$(echo "$KERNEL_REGEX" | sed 's/[^a-zA-Z0-9_].*//' | tr '[:upper:]' '[:lower:]')"
else
  KFILTER=""
  KARG=""
  KERNEL_SLUG="all"
fi

OUT_BASE="$OUT_DIR/${REP_NAME}_${KERNEL_SLUG}"

echo "[ncu] Dumping SASS from $REP (kernel: ${KERNEL_REGEX:-all}) -> ${OUT_BASE}.sass" >&2
ncu -i "$REP_PATH" $KFILTER $KARG --page source --print-source sass 2>&1 | tee "${OUT_BASE}.sass" > /dev/null

echo "[ncu] Dumping PTX -> ${OUT_BASE}.ptx" >&2
ncu -i "$REP_PATH" $KFILTER $KARG --page source --print-source ptx  2>&1 | tee "${OUT_BASE}.ptx"  > /dev/null

echo "Wrote ${OUT_BASE}.sass and ${OUT_BASE}.ptx"
