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

# --- CUDA 12.8 (skip if already installed) ---
if ! /usr/local/cuda-12.8/bin/nvcc --version &>/dev/null; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
  sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
  sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda-toolkit-12-8
fi
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- venv ---
uv sync
source .venv/bin/activate

# --- arch detection (CUDA_ARCH env var wins; otherwise query nvidia-smi) ---
if [ -z "${CUDA_ARCH:-}" ]; then
  CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | tr -d ' ' | sed 's/\.//')"
  CUDA_ARCH="sm_${CUDA_ARCH}"
fi
export CUDA_ARCH
echo "[benchmark.sh] arch=${CUDA_ARCH}"

# --- generate kernel_config.h for this arch, then build ---
python scripts/gen_kernel_config.py "${CUDA_ARCH}"
# Avoid PEP 517 isolation reinstalling PyTorch (huge); use venv's torch for the extension build
uv pip install -e . --no-build-isolation

# --- (optional) autotune: set AUTOTUNE=1 to run before benchmarking ---
if [ "${AUTOTUNE:-0}" = "1" ]; then
  echo "[benchmark.sh] running autotune (AUTOTUNE=1)"
  python scripts/autotune.py --arch "${CUDA_ARCH}" ${AUTOTUNE_ARGS:-}
  # Rebuild with the tuned config
  python scripts/gen_kernel_config.py "${CUDA_ARCH}"
  uv pip install -e . --no-build-isolation
fi

python benchmark.py
