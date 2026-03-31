# Softmax

the write up for these is at :
- https://www.dhrv.org/blog/04/softmax-1/
- https://www.dhrv.org/blog/04/softmax-2/


## Running the Benchmark

### Default run (auto-detect arch)

```bash
./scripts/benchmark.sh
```

### Specify arch explicitly

Set `CUDA_ARCH` to override auto-detection. Supported archs: `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`.

```bash
CUDA_ARCH=sm_89 ./scripts/benchmark.sh
```

Or persist it in your shell / `.env`:

```bash
export CUDA_ARCH=sm_89
./scripts/benchmark.sh
```

### Run with autotune

Set `AUTOTUNE=1` to run coordinate-descent tuning before benchmarking. The tuner tries candidate values for register count and per-kernel launch parameters, rebuilds the extension for each candidate, and writes the best config back to `configs/archs/<arch>.yml` before the final benchmark run.

```bash
AUTOTUNE=1 ./scripts/benchmark.sh
```

With an explicit arch:

```bash
CUDA_ARCH=sm_89 AUTOTUNE=1 ./scripts/benchmark.sh
```

Pass extra flags to the tuner via `AUTOTUNE_ARGS`:

```bash
# Tune only the online_v2 kernel, limit to 2 coordinate-descent rounds
AUTOTUNE=1 AUTOTUNE_ARGS="--kernel online_v2 --rounds 2" ./scripts/benchmark.sh
```

#### Run autotune standalone

```bash
# Auto-detect arch, tune all kernels
python scripts/autotune.py

# Explicit arch
python scripts/autotune.py --arch sm_89

# Single kernel (fused_warp | fused_block | online | online_v2)
python scripts/autotune.py --arch sm_89 --kernel online_v2

# Dry-run: print what would be tried without rebuilding
python scripts/autotune.py --arch sm_89 --dry-run

# Limit coordinate-descent rounds
python scripts/autotune.py --arch sm_89 --rounds 2
```

Tunable parameters and their search spaces:

| Parameter | Candidates |
|---|---|
| `compile.maxrregcount` | 64, 96, 128 |
| `kernels.online.two_pass_min_np` | 8, 16, 32 |
| `kernels.online_v2.multi_block.target_np` | 4, 8, 16 |
| `kernels.online_v2.multi_block.split_threshold` | 8, 16, 32 |

After autotuning, the winning config is saved to `configs/archs/<arch>.yml`. 

### Add a new arch config

If your GPU arch isn’t in `configs/archs/` (e.g. `sm_87`), copy the closest existing config and edit it:

```bash
cp configs/archs/sm_86.yml configs/archs/sm_87.yml
# edit configs/archs/sm_87.yml as needed
python scripts/gen_kernel_config.py sm_87
uv pip install -e . --no-build-isolation
python benchmark.py
```

---

## Scripts

### `run_ncu_collect.sh`

Profiles all softmax variants with NVIDIA Nsight Compute and writes `.ncu-rep` reports into `out/`.

**Usage:**

```bash
./run_ncu_collect.sh [N]
```
- `N` — problem size for profiling (default: 4096; use 8192 to force block kernel for fused).

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
