#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arch_config import detect_arch, load_config  # noqa: E402


def _macro_line(name: str, val) -> str:
    return f"#define {name} {val}"


def _sizes_macro(macro_name: str, sizes: list, one_arg: bool = False) -> list[str]:
    """Build a multi-line X-macro definition."""
    lines = [f"#define {macro_name}(F)  \\"]
    for i, entry in enumerate(sizes):
        comma = "\\" if i < len(sizes) - 1 else ""
        if one_arg:
            lines.append(f"  F({entry}){' ' + comma if comma else ''}")
        else:
            n, bs = entry
            lines.append(f"  F({n}, {bs}){' ' + comma if comma else ''}")
    return lines


def generate_header(cfg: dict) -> str:
    arch = cfg["arch"]
    kernels = cfg["kernels"]
    fw = kernels["fused_warp"]
    fb = kernels["fused_block"]
    ol = kernels["online"]
    v2 = kernels["online_v2"]
    v2s = v2["single_block"]
    v2m = v2["multi_block"]

    # Derived range bounds
    fw_min = min(fw["supported_n"])
    fw_max = max(fw["supported_n"])
    fb_min = min(fb["supported_n"])
    fb_max = max(fb["supported_n"])
    ol_min = min(n for n, _ in ol["sizes"])
    ol_max = max(n for n, _ in ol["sizes"])
    v2_min = min(n for n, _ in v2s["sizes"])
    v2_max_overall = max(
        max(n for n, _ in v2s["sizes"]),
        ol_max,  # multi-block covers up to the same range as online
    )

    lines: list[str] = []
    lines += [
        "// AUTO-GENERATED — do not edit.",
        f"// Source: configs/archs/{arch}.yml",
        "// Re-generate: python scripts/gen_kernel_config.py",
        "",
        "#ifndef SOFTMAX_KERNEL_CONFIG_H",
        "#define SOFTMAX_KERNEL_CONFIG_H",
        "",
        "// ---------------------------------------------------------------------------",
        "// fused_warp kernel",
        "// ---------------------------------------------------------------------------",
        _macro_line("FUSED_WARP_THREADBLOCK_SIZE", fw["threadblock_size"]),
        _macro_line("FUSED_WARP_MIN_N", fw_min),
        _macro_line("FUSED_WARP_MAX_N", fw_max),
        "",
    ]
    lines += _sizes_macro("FUSED_WARP_SIZES", fw["supported_n"], one_arg=True)
    lines += [
        "",
        "// ---------------------------------------------------------------------------",
        "// fused_block kernel",
        "// ---------------------------------------------------------------------------",
        _macro_line("FUSED_BLOCK_SIZE", fb["block_size"]),
        _macro_line("FUSED_BLOCK_MIN_N", fb_min),
        _macro_line("FUSED_BLOCK_MAX_N", fb_max),
        "",
    ]
    lines += _sizes_macro("FUSED_BLOCK_SIZES", fb["supported_n"], one_arg=True)
    lines += [
        "",
        "// ---------------------------------------------------------------------------",
        "// online kernel",
        "// ---------------------------------------------------------------------------",
        _macro_line("ONLINE_MIN_N", ol_min),
        _macro_line("ONLINE_MAX_N", ol_max),
        _macro_line("ONLINE_2PASS_MIN_NP", ol["two_pass_min_np"]),
        "",
    ]
    lines += _sizes_macro("ONLINE_SIZES", ol["sizes"])
    lines += [
        "",
        "// ---------------------------------------------------------------------------",
        "// online_v2 kernel",
        "// ---------------------------------------------------------------------------",
        _macro_line("ONLINE_V2_MIN_N", v2_min),
        _macro_line("ONLINE_V2_MAX_N", v2_max_overall),
        _macro_line("ONLINE_V2_SINGLE_MAX_N", v2s["max_n"]),
        "",
    ]
    lines += _sizes_macro("ONLINE_V2_SINGLE_SIZES", v2s["sizes"])
    lines += [
        "",
        _macro_line("V2_TARGET_NP", v2m["target_np"]),
        _macro_line("V2_MULTI_BS", v2m["block_size"]),
        _macro_line("V2_SPLIT_THRESHOLD", v2m["split_threshold"]),
        "",
        "#endif /* SOFTMAX_KERNEL_CONFIG_H */",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import os  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", nargs="?", help="e.g. sm_80 (default: auto-detect)")
    args = parser.parse_args()

    arch = args.arch or os.environ.get("CUDA_ARCH") or detect_arch()
    cfg = load_config(arch)
    header = generate_header(cfg)

    out_path = ROOT / "csrc" / "kernel_config.h"
    out_path.write_text(header)
    print(f"[gen_kernel_config] wrote {out_path} for arch={arch}")


if __name__ == "__main__":
    main()
