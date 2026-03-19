#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# --- uv (if not already available) ---
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# Ensure uv is on PATH when we run it (e.g. after install in same run)
export PATH="${HOME}/.local/bin:${PATH:-}"

# --- CUDA 12.8 (nvcc in PATH may be wrong version; require toolkit at expected path) ---
if [ ! -d /usr/local/cuda-12.8 ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
  sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
  sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda-toolkit-12-8
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- venv and benchmark ---
export CUDA_ARCH="${CUDA_ARCH:-sm_75}"
uv sync
source .venv/bin/activate
uv pip install -e .
python benchmark.py
