#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
DIR="${1:-$ROOT/out}"
mkdir -p "$DIR"

for rep in "$DIR"/*.ncu-rep; do
  [ -f "$rep" ] || continue
  base="${rep%.ncu-rep}"
  txt="${base}.ncu-rep.txt"
  echo "[ncu] Exporting $(basename "$rep") -> $(basename "$txt")"
  ncu -i "$rep" --page details 2>&1 | tee "$txt" > /dev/null
done

echo "Text exports: $DIR/*.ncu-rep.txt"
