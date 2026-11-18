# RandomizedTensorCompressionCXX
CXX implementations of Star-M tensor compression algorithms.

Install Just and Visual Studio 2022 or 2026 with **Desktop development with C++**,
a Windows SDK, and CMake 3.25 or newer.

Then run the following commands from a PowerShell prompt in the project root directory:

```powershell
just run                 # Build and run optimized code with PDB debug information
just nocompile           # Run the existing RelWithDebInfo executable without building
just debug               # Build and run unoptimized Debug code
just build               # Incrementally build RelWithDebInfo without running
just configure           # Generate the Windows x64 build tree
just fetch-deps          # Download, build, and install the dependencies locally
just clean               # Clean the application, retaining built dependencies
just run "D:/other data" # Use this directory for the NetCDF experiment
```

At the menu, enter experiment numbers separated by spaces. For example,
`2 1 2` runs experiment 2, then 1, then 2 again. Unknown numbers are skipped.
Press Enter on a blank line to quit.

## Measured experiments

Run the Jungtelziemniak2d and Boussinesq benchmarks from the project root with
Python, NumPy, pandas, and Matplotlib installed.

```powershell
python experiments/measuredExperiments/run_measured.py
```

The runner builds the optimized C++ executable, runs both planar datasets,
and looks for a shared MATLAB session. To include live MATLAB timings, install
the MATLAB Engine for your Python and MATLAB release. In MATLAB, put TeBaC
on the path, choose the parallel pool you want to measure, and call
`matlab.engine.shareEngine`. The experiments always time the serial code and
also time the parallel code when a pool is open. Each pool size gets its own
CSV, so you can run the MATLAB functions again with a different pool.

If you already have MATLAB CSVs, use `--matlab csv` to run C++ and collect
them. The default also collects existing CSVs when no engine or shared
session is available. Use `--session NAME` to select a particular shared
session. Use `--plot-only` to rebuild the figures from completed CSVs.

| Dataset | Tensor constructed from full-resolution `u` | C++ scaling | C++ scheduler | MATLAB function |
| --- | --- | ---: | ---: | --- |
| Jungtelziemniak2d | 200 × 500 × 450 | 11 | 14 | `measureJungtelziemniak2d` |
| Boussinesq | 450 × 2001 × 150 | 12 | 15 | `measureBoussinesq` |
| Tangaroa | 54000 × 201 × 120 | 13 | 16 | `measureTangaroa` |

Tangaroa is manual. Enter `13 16` at the `just run` menu for C++. In MATLAB,
with TeBaC on the path, run the following from this project root. The next
plotting run will pick up the completed Tangaroa CSVs.

```matlab
addpath(fullfile(pwd, 'experiments', 'measuredExperiments'))
measureTangaroa()
```

The planar tensors follow TeBaC's height × time × width layout. Tangaroa
flattens each full x-y plane into one column, with time along the second
axis and depth along the third. Each timed call includes the mode-3 DCT,
the economy slice SVDs, and truncation, and returns transform-domain factors.
File reads, construction of the DCT matrix, pool setup, and CSV writes occur
outside the timer. MATLAB's serial baseline uses an ordinary loop over slices.
Its numerical libraries retain their own internal threading behavior.
The MATLAB experiments use TeBaC's slice SVDs. A local truncation helper
takes the row counts from the economy U and V factors so it can handle
rectangular slices. The serial TSVDM2 baseline applies TeBaC's global energy
cutoff to the output of `TSVDMserial`.

- **TSVDM** gives every slice the same rank `c` and measures every integer
  from 1 through the smaller frontal-slice dimension, as in TeBaC's
  `compressionComparison.m`.
- **TSVDM2** measures retained energy fractions 0.9, 0.99, 0.999, and 0.9999.
- **C++ scaling** uses the serial loop and then 2, 4, 8, 16, and 32 threads,
  with one slice per scheduler batch.
- **C++ scheduler** uses 32 threads and batches of 1, 2, 4, 8, 16, and 32
  slices, with `c = 8` for TSVDM and `gamma = 0.99` for TSVDM2.

Results go in `build/measuredExperiments/`. Each configuration contributes
five wall-clock timings to `timings.csv`. The means, speedups, and efficiencies
go in `summary.csv`, and the runtime, speedup, and efficiency plots go in
`plots/` as PNG and PDF files. Speedup is the matching serial mean divided
by the parallel mean, using the same dataset, language, algorithm, and
truncation setting. Efficiency is 100 times speedup divided by the thread
or worker count. The scheduler plots use the same serial baseline.

The runner accepts `--data-dir` and `--output`. Each MATLAB function accepts
the same directories as optional positional arguments. For manual C++ runs,
pass the data directory to Just and set `RTC_MEASURED_OUTPUT` to select the
output directory. A completed run replaces the CSV for that dataset,
language, sweep, and MATLAB pool size. Use a separate output directory to
keep results from different machines or runs.
