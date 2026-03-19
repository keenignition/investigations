#!/usr/bin/env python3
"""
Compare NCU report files: write key metrics and summary to a CSV file.

Usage:
  python ncu_compare.py [directory]             # directory with .ncu-rep files (default: out)
  python ncu_compare.py --from-txt [directory]   # parse existing .ncu-rep.txt files (faster)
  python ncu_compare.py -o path.csv [directory] # write CSV to path.csv (default: <directory>/ncu_compare.csv)

If --from-txt is not set, runs `ncu -i <file> --page details` for each .ncu-rep (slow).
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


# Skip kernels that are from PyTorch/at:: (init, etc.)
SKIP_KERNEL_PATTERN = re.compile(r"void at::|at::<unnamed>")

# Key metrics to compare (lower is better for Duration, higher better for throughput %)
METRICS_LOWER_BETTER = {"Duration", "Elapsed Cycles"}
KEY_METRICS = [
    "Duration",
    "Elapsed Cycles",
    "Memory Throughput",
    "DRAM Throughput",
    "Compute (SM) Throughput",
    "SM Busy",
    "Max Bandwidth",
    "L2 Hit Rate",
    "Executed Ipc Elapsed",
]


def parse_details_text(text: str) -> list[dict[str, dict[str, str]]]:
    """Parse ncu --page details output. Returns list of kernels, each a dict of section -> {metric_name: value}."""
    kernels = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Kernel header: line with "Context" and "Stream" (kernel signature)
        if "Context" in line and "Stream" in line and "Device" in line:
            if SKIP_KERNEL_PATTERN.search(line):
                i += 1
                continue
            # New kernel block
            kernel_metrics: dict[str, str] = {}
            i += 1
            while i < len(lines):
                l = lines[i]
                if l.strip().startswith("Section:"):
                    section = l.replace("Section:", "").strip()
                    i += 1
                    # Skip table header line (Metric Name ... Metric Unit ... Metric Value)
                    if i < len(lines) and ("Metric Name" in lines[i] or "----" in lines[i]):
                        i += 1
                    if i < len(lines) and "----" in lines[i]:
                        i += 1
                    # Parse table rows: "Name    Unit    Value" (variable spacing)
                    while i < len(lines):
                        row = lines[i]
                        if not row.strip() or row.strip().startswith("Section:") or row.strip().startswith("INF") or row.strip().startswith("OPT"):
                            break
                        if "----" in row:
                            i += 1
                            continue
                        # Three columns: split from the end (value, unit, name)
                        parts = row.split()
                        if len(parts) >= 2:
                            value = parts[-1].rstrip(",")
                            # Usually "Name   unit   value"; unit often %, ms, Ghz, cycle, etc.
                            units = {"%", "Ghz", "cycle", "us", "ms", "Kbyte", "warp", "inst/cycle", "inst"}
                            if len(parts) >= 3 and parts[-2] in units:
                                name = " ".join(parts[:-2]).strip()
                            else:
                                name = " ".join(parts[:-1]).strip()
                            if name and name != "Metric Name" and not name.startswith("---"):
                                kernel_metrics[name] = value
                        i += 1
                    continue
                if l.strip() and "Context" in l and "Stream" in l:
                    break  # Next kernel
                i += 1
            if kernel_metrics:
                kernels.append(kernel_metrics)
            continue
        i += 1
    return kernels


def get_first_kernel_metrics(metrics_list: list[dict[str, str]]) -> dict[str, str] | None:
    return metrics_list[0] if metrics_list else None


def run_ncu_import(report_path: Path) -> str:
    result = subprocess.run(
        ["ncu", "-i", str(report_path), "--page", "details"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ncu failed: {result.stderr or result.stdout}")
    return result.stdout + result.stderr


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--from-txt"]
    from_txt = "--from-txt" in sys.argv
    out_path: Path | None = None
    if "-o" in args:
        i = args.index("-o")
        if i + 1 < len(args):
            out_path = Path(args[i + 1]).resolve()
            args.pop(i)
            args.pop(i)
    dir_arg = args[0] if args else "out"
    root = Path(__file__).resolve().parent
    dir_path = (root / dir_arg).resolve()
    if not dir_path.is_dir():
        print(f"Directory not found: {dir_path}", file=sys.stderr)
        sys.exit(1)

    rep_files = sorted(dir_path.glob("*.ncu-rep"))
    if not rep_files:
        print(f"No .ncu-rep files in {dir_path}", file=sys.stderr)
        sys.exit(1)

    # Collect metrics per report
    name_to_metrics: dict[str, dict[str, str]] = {}
    for rep in rep_files:
        name = rep.stem
        if from_txt:
            txt = dir_path / f"{name}.ncu-rep.txt"
            if not txt.exists():
                print(f"Missing {txt}; run without --from-txt or run ncu_export_txt.sh first.", file=sys.stderr)
                continue
            text = txt.read_text()
        else:
            print(f"Importing {rep.name} ...", file=sys.stderr)
            text = run_ncu_import(rep)
        kernels = parse_details_text(text)
        metrics = get_first_kernel_metrics(kernels)
        if metrics:
            name_to_metrics[name] = metrics
        else:
            print(f"No kernel metrics found in {rep.name}", file=sys.stderr)

    if not name_to_metrics:
        print("No metrics collected.", file=sys.stderr)
        sys.exit(1)

    # Build comparison table
    all_metric_names: set[str] = set()
    for m in name_to_metrics.values():
        all_metric_names.update(m.keys())
    # Use key metrics first, then the rest; skip junk rows
    skip = {"Metric Name Metric Unit Metric", "Metric Name"}
    ordered = [x for x in KEY_METRICS if x in all_metric_names]
    for k in sorted(all_metric_names):
        if k not in ordered and k not in skip:
            ordered.append(k)

    names = list(name_to_metrics.keys())

    def norm_val(v: str) -> float | None:
        v = v.replace(",", "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    csv_path = out_path or (dir_path / "ncu_compare.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Main table: metric, report1, report2, ..., best
        w.writerow(["metric"] + names + ["best"])
        for metric in ordered:
            row_vals = [name_to_metrics[n].get(metric, "") for n in names]
            numeric = [norm_val(v) for v in row_vals]
            best = ""
            if all(x is not None for x in numeric) and len(numeric) > 1:
                if metric in METRICS_LOWER_BETTER:
                    best_idx = min(range(len(numeric)), key=lambda i: numeric[i])
                else:
                    best_idx = max(range(len(numeric)), key=lambda i: numeric[i])
                best = names[best_idx]
            w.writerow([metric] + row_vals + [best])

        # Duration ranking
        w.writerow([])
        w.writerow(["duration_rank", "report", "duration"])
        if "Duration" in ordered:
            by_duration = []
            for n in names:
                v = name_to_metrics[n].get("Duration", "")
                num = norm_val(v)
                if num is not None:
                    by_duration.append((n, num))
            by_duration.sort(key=lambda x: x[1])
            for i, (n, d) in enumerate(by_duration, 1):
                w.writerow([i, n, d])
            if len(by_duration) > 1 and by_duration[-1][1] > 0:
                speedup = by_duration[-1][1] / by_duration[0][1]
                w.writerow(["", "speedup (worst/best)", f"{speedup:.2f}x"])

        # Suggestions
        w.writerow([])
        w.writerow(["report", "suggestion"])
        for n in names:
            m = name_to_metrics[n]
            tips = []
            sm_busy = norm_val(m.get("SM Busy", ""))
            mem_thr = norm_val(m.get("Memory Throughput", ""))
            if sm_busy is not None and sm_busy < 30 and mem_thr is not None and mem_thr < 50:
                tips.append("Low SM and memory utilization → consider memory-bound optimizations or larger workload.")
            elif mem_thr is not None and mem_thr > 80:
                tips.append("Memory-bound (high DRAM throughput) → optimize memory access pattern, coalescing, or use shared memory.")
            if m.get("Duration"):
                tips.append(f"Duration {m['Duration']} (compare with others above).")
            if tips:
                w.writerow([n, " ".join(tips)])

    print(f"Wrote {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
