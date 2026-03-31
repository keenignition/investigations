from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def detect_arch() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0].strip()
        major, minor = out.split(".")
        return f"sm_{major}{minor}"
    except Exception:
        pass
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            maj, minor = torch.cuda.get_device_capability(0)
            return f"sm_{maj}{minor}"
    except Exception:
        pass
    return "sm_75"


def load_config(arch: str | None = None) -> dict:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("PyYAML is required: pip install pyyaml") from e

    if arch is None:
        arch = os.environ.get("CUDA_ARCH") or detect_arch()

    cfg_path = _ROOT / "configs" / "archs" / f"{arch}.yml"
    if not cfg_path.exists():
        available = sorted(p.stem for p in cfg_path.parent.glob("*.yml"))
        raise FileNotFoundError(
            f"No config for {arch!r}. Available: {available}"
        )
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_arch"] = arch
    return cfg
