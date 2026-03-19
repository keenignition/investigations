#!/usr/bin/env bash
# Fast editable install: skips PEP 517 isolated build env (which would reinstall PyTorch).
# Prerequisites: deps already in the active venv (e.g. `uv sync`).
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v uv >/dev/null 2>&1; then
  exec uv pip install -e . --no-build-isolation "$@"
else
  exec pip install -e . --no-build-isolation "$@"
fi
