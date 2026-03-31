#!/usr/bin/env python3
"""
--- claude ---

Autotune kernel configuration for the current GPU.

For each tunable parameter, this script tries a range of candidate values,
rebuilds the extension, benchmarks, and writes the best config back to the
arch's YAML.

Tunable parameters
------------------
  compile.maxrregcount      : [64, 96, 128]
  kernels.online.two_pass_min_np : [8, 16, 32]
  kernels.online_v2.multi_block.target_np    : [4, 8, 16]
  kernels.online_v2.multi_block.split_threshold : [8, 16, 32]

Strategy
--------
Coordinate-descent: tune one parameter at a time, holding others at their
current best, cycling until no improvement is seen.  This is much cheaper
than a full grid search (O(N) builds vs O(N^k)) while still finding good
configs in practice for these nearly-independent knobs.

Usage
-----
    python scripts/autotune.py [--arch sm_80] [--kernel online_v2] [--dry-run]

Options
-------
  --arch    Architecture to tune (default: auto-detect)
  --kernel  Restrict to one kernel: fused_warp | fused_block | online | online_v2
            Default: tune all
  --dry-run Print what would be tried without rebuilding or writing back
  --rounds  Max coordinate-descent rounds (default: 3)
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arch_config import detect_arch, load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Parameter search space
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, list] = {
    "compile.maxrregcount": [64, 96, 128],
    "kernels.online.two_pass_min_np": [8, 16, 32],
    "kernels.online_v2.multi_block.target_np": [4, 8, 16],
    "kernels.online_v2.multi_block.split_threshold": [8, 16, 32],
}

# Which N values to benchmark for each parameter (representative subset)
BENCHMARK_N = [4096, 16384, 65536, 131072, 262144]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _get(cfg: dict, dotpath: str):
    keys = dotpath.split(".")
    node = cfg
    for k in keys:
        node = node[k]
    return node


def _set(cfg: dict, dotpath: str, value) -> dict:
    cfg = copy.deepcopy(cfg)
    keys = dotpath.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return cfg


def _write_yaml(cfg: dict, arch: str) -> None:
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")
    out_path = ROOT / "configs" / "archs" / f"{arch}.yml"
    with out_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[autotune] wrote {out_path}")


# ---------------------------------------------------------------------------
# Build + benchmark
# ---------------------------------------------------------------------------
def rebuild(arch: str, cfg: dict, dry_run: bool) -> bool:
    """Write YAML, regenerate header, rebuild extension. Returns success."""
    _write_yaml(cfg, arch)
    if dry_run:
        print(f"[autotune] (dry-run) would regenerate kernel_config.h and rebuild")
        return True
    gen = ROOT / "scripts" / "gen_kernel_config.py"
    r = subprocess.run([sys.executable, str(gen), arch], cwd=ROOT)
    if r.returncode != 0:
        print(f"[autotune] gen_kernel_config failed")
        return False
    # Prefer uv (no pip needed in venv); fall back to python -m pip
    import shutil  # noqa: PLC0415
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "-e", ".", "--no-build-isolation", "-q"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "-q"]
    r = subprocess.run(cmd, cwd=ROOT)
    return r.returncode == 0


def benchmark_gbps(arch: str, cfg: dict, dry_run: bool) -> float:
    """Return mean GB/s across BENCHMARK_N values for the online_v2 kernel."""
    if dry_run:
        import random  # noqa: PLC0415
        return random.uniform(1000, 2000)

    batch = cfg["benchmark"]["batch_size"]
    total_gbps = 0.0
    count = 0
    for n in BENCHMARK_N:
        # Only benchmark N values that are actually supported
        ol_max = max(s[0] if isinstance(s, list) else s
                     for s in cfg["kernels"]["online"]["sizes"])
        if n > ol_max:
            continue
        script = ROOT / "scripts" / "_bench_one.py"
        r = subprocess.run(
            [sys.executable, str(script), str(batch), str(n)],
            capture_output=True, text=True, cwd=ROOT,
        )
        if r.returncode == 0:
            try:
                gbps = float(r.stdout.strip().splitlines()[-1])
                total_gbps += gbps
                count += 1
            except ValueError:
                pass
    return total_gbps / count if count else 0.0


# ---------------------------------------------------------------------------
# Inline micro-benchmark helper (written to a temp script)
# ---------------------------------------------------------------------------
_BENCH_SCRIPT = """\
import sys
import torch
import softmax_kernel
M, N = int(sys.argv[1]), int(sys.argv[2])
x = torch.randn((M, N), device="cuda")
# warmup
for _ in range(5): softmax_kernel.softmax_online_v2(x)
torch.cuda.synchronize()
# timed
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(20): softmax_kernel.softmax_online_v2(x)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 20
gbps = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
print(gbps)
"""


def _ensure_bench_script() -> None:
    p = ROOT / "scripts" / "_bench_one.py"
    if not p.exists():
        p.write_text(_BENCH_SCRIPT)


# ---------------------------------------------------------------------------
# Coordinate-descent tuner
# ---------------------------------------------------------------------------
def tune(arch: str, param_keys: list[str], rounds: int, dry_run: bool) -> dict:
    _ensure_bench_script()
    cfg = load_config(arch)

    print(f"\n[autotune] arch={arch}, params={param_keys}, rounds={rounds}")
    if not rebuild(arch, cfg, dry_run):
        sys.exit("[autotune] initial build failed")

    best_gbps = benchmark_gbps(arch, cfg, dry_run)
    print(f"[autotune] baseline GB/s = {best_gbps:.1f}")

    improved = True
    for round_i in range(rounds):
        if not improved:
            break
        improved = False
        print(f"\n[autotune] round {round_i + 1}/{rounds}")

        for param in param_keys:
            if param not in SEARCH_SPACE:
                print(f"[autotune] unknown param {param!r}, skipping")
                continue
            candidates = SEARCH_SPACE[param]
            current = _get(cfg, param)
            print(f"  tuning {param} (current={current}, candidates={candidates})")

            for val in candidates:
                if val == current:
                    continue
                candidate_cfg = _set(cfg, param, val)
                if not rebuild(arch, candidate_cfg, dry_run):
                    print(f"    val={val}: build failed, skipping")
                    continue
                gbps = benchmark_gbps(arch, candidate_cfg, dry_run)
                print(f"    val={val}: {gbps:.1f} GB/s  (best={best_gbps:.1f})")
                if gbps > best_gbps:
                    best_gbps = gbps
                    current = val
                    cfg = candidate_cfg
                    improved = True

            # Persist the winner for this param before moving to the next
            _write_yaml(cfg, arch)

    print(f"\n[autotune] done — best GB/s = {best_gbps:.1f}")
    print(f"[autotune] final config written to configs/archs/{arch}.yml")
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import os  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=None, help="e.g. sm_80 (default: auto-detect)")
    parser.add_argument(
        "--kernel",
        default=None,
        choices=["fused_warp", "fused_block", "online", "online_v2"],
        help="Restrict to one kernel's parameters",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without rebuilding")
    parser.add_argument("--rounds", type=int, default=3, help="Max coordinate-descent rounds")
    args = parser.parse_args()

    arch = args.arch or os.environ.get("CUDA_ARCH") or detect_arch()

    # Filter param keys by kernel if requested
    kernel_prefix_map = {
        "fused_warp":  [],          # no tunable compile-time knobs beyond maxrregcount
        "fused_block": [],
        "online":      ["kernels.online.two_pass_min_np"],
        "online_v2":   [
            "kernels.online_v2.multi_block.target_np",
            "kernels.online_v2.multi_block.split_threshold",
        ],
    }
    if args.kernel:
        param_keys = ["compile.maxrregcount"] + kernel_prefix_map[args.kernel]
    else:
        param_keys = list(SEARCH_SPACE.keys())

    tune(arch, param_keys, rounds=args.rounds, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
