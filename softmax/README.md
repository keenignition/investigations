# Softmax

This project needs CUDA/nvcc 12.8.

## Setup

1. Copy `.env.example` to `.env` and set `CUDA_ARCH` for your GPU (e.g. `sm_89`).
2. Run the benchmark script (creates `.venv`, `uv sync`, then fast editable install + `benchmark.py`):

   ```bash
   source .env   # or: export $(grep -v '^#' .env | xargs)
   ./scripts/benchmark.sh
   ```

   The script uses `uv pip install -e . --no-build-isolation` so the CUDA extension builds against the venv’s PyTorch instead of a fresh isolated multi‑GB torch install.

**Manual install** (if not using `benchmark.sh`):

   ```bash
   uv sync
   source .venv/bin/activate
   uv pip install -e . --no-build-isolation
   ```

### Why `pip install -e .` can take ~10 minutes (especially on a remote box)

`pip` uses **PEP 517 build isolation** by default. Your `[build-system] requires` includes **`torch`**, so each install may create a **fresh build environment** and **download/extract a full PyTorch wheel** (gigabytes) *before* nvcc touches your tiny `.cu` files. Kernel size does not matter much; that step dominates.

**Fast iterative workflow** (use the venv’s torch instead of reinstalling it for every build): same as `./scripts/benchmark.sh` after `uv sync`, or:

```bash
uv sync   # once: pulls torch + ninja into .venv
uv pip install -e . --no-build-isolation
# or:  pip install -e . --no-build-isolation
```

To see where time goes: `pip install -e . -v` (watch for “Installing build dependencies” / torch).

**Optional compile speedups** (after the isolation issue is fixed):

- **Parallel nvcc**: Ninja is used by default if `ninja` is on `PATH`; cap jobs with `export MAX_JOBS=8`.
- **ccache**: `export CCACHE_DIR=~/.ccache` and wrap `nvcc` / `c++` with ccache for faster rebuilds.
- **Profiling-oriented flags**: `setup.py` passes `-Xptxas=-v` and `-lineinfo`; fine for analysis, slightly more work than a minimal dev build—only matters once PyTorch isn’t being reinstalled every time.

---

## Scripts

### `run_ncu_collect.sh`

Profiles all softmax variants with NVIDIA Nsight Compute and writes `.ncu-rep` reports into `out/`.

**Usage:**

```bash
./run_ncu_collect.sh [N]
```

- `N` — problem size for profiling (default: 4096; use 8192 to force block kernel for fused).
- Output: `out/softmax_naive.ncu-rep`, `out/softmax_wr.ncu-rep`, `out/softmax_fused.ncu-rep`, `out/softmax_block.ncu-rep`, `out/fused_triton.ncu-rep`.

### `ncu_export_txt.sh`

Exports `.ncu-rep` files to text so you can diff or parse them. Running `ncu -i` per report is slow; exporting once then using `ncu_compare.py --from-txt` is faster for repeated comparisons.

**Usage:**

```bash
./ncu_export_txt.sh [directory]
```

- `directory` — directory containing `.ncu-rep` files (default: `out/`).
- Output: `directory/<name>.ncu-rep.txt` for each `<name>.ncu-rep`.

### `ncu_compare.py`

Compares NCU reports and writes key metrics, duration ranking, and suggestions to a CSV file.

**Usage:**

```bash
# Compare using existing text exports (fast); writes <directory>/ncu_compare.csv
python ncu_compare.py --from-txt [directory]

# Compare by reading .ncu-rep directly (slower, no export step)
python ncu_compare.py [directory]

# Write CSV to a specific path
python ncu_compare.py -o path/to/compare.csv [directory]
```

- `directory` — directory with `.ncu-rep` (and optionally `.ncu-rep.txt`) files (default: `out`).
- `-o path` — output CSV path (default: `<directory>/ncu_compare.csv`).

The script uses the first non–PyTorch kernel in each report. The CSV contains: a metric table with one column per report and a `best` column; a duration ranking section; and a suggestions section. A single line (e.g. `Wrote out/ncu_compare.csv`) is printed to stderr.

### `ncu_dump_sass.sh`

Dumps SASS and PTX for a kernel from an NCU report to text files (for grepping, diffing, or scripting).

**Usage:**

```bash
./ncu_dump_sass.sh <report.ncu-rep> [kernel_regex] [output_dir]
```

- `report.ncu-rep` — path to the report.
- `kernel_regex` — optional; `-k` filter for kernel name (e.g. `"softmax_kernel"`). If omitted, all kernels are dumped.
- `output_dir` — optional (default: `out/ncu_dumps`).

Output: `<output_dir>/<report_stem>_<kernel_slug>.sass` and `.ptx`. Example:  
`./ncu_dump_sass.sh out/softmax_naive.ncu-rep "softmax_kernel" out/dumps`

---

## Comparing NCU reports

1. **Collect reports** (if you don’t have them yet):

   ```bash
   ./run_ncu_collect.sh
   ```

2. **Export to text** (optional; makes later comparison faster):

   ```bash
   ./ncu_export_txt.sh out
   ```

3. **Compare and open the CSV:**

   ```bash
   python ncu_compare.py --from-txt out
   ```

   Or without exporting first:

   ```bash
   python ncu_compare.py out
   ```

   Output is written to `out/ncu_compare.csv` (or the path given with `-o`). The CSV has the metric table (with a `best` column per row), duration ranking, and suggestions.

**Other ways to inspect reports:**

- Full text for one report:  
  `ncu -i out/softmax_block.ncu-rep --page details`
- Single kernel by name:  
  `ncu -i out/softmax_naive.ncu-rep -k "softmax_kernel" --page details`
- CSV (for scripting):  
  `ncu -i out/softmax_naive.ncu-rep --csv --page raw`
- Open in Nsight Compute UI:  
  `ncu-ui out/softmax_block.ncu-rep` (if the UI is installed)
