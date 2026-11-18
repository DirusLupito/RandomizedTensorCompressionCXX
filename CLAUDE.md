# CLAUDE.md

## What this project is

Final project for CSC 548 (Parallel Computing), done by two students. The
assignment was open-ended: pick a problem and solve it with parallel computing.

We are porting two tensor compression algorithms from MATLAB to C++23 and
comparing their performance, single-threaded and multithreaded. The algorithms
come from a research lab paper that one of us co-authored (Kilmer et al. 2021,
"Tensor-tensor algebra for optimal representation and compression of multiway
data"). The question we're answering is whether MATLAB's abstractions
noticeably slow these algorithms down, or whether the cost is amortized away.

- **Algorithm 1, star-M SVD I (`TSVDM`)**: transform the tensor along mode 3 by a
  unitary matrix `M`, take a rank-`k` truncated SVD of every frontal slice, then
  transform the factors back with `M^T`.
- **Algorithm 2, star-M SVD II (`TSVDM2`)**: compute the full star-M SVD, then
  choose a per-slice rank `rho_i` from a global energy threshold `gamma`. All
  squared singular values are pooled, sorted, and cumulatively summed.

`proposal.tex` has the full problem statement and pseudocode. Ignore its
milestone dates, since the timeline has changed.

## Where it's going: parallelism is the priority

The proposal feedback was that the parallel component was weak and that the
project looked like MATLAB vs. C++ rather than a parallel computing project.
**The presentation and report must emphasize the parallel work.** When choosing
between designs, prefer the ones that give us more parallel substance to
implement, measure, and discuss. Examples:

- The per-slice SVD loop, the C++ equivalent of MATLAB's `parfor`. This is the
  obvious first target. Slices are independent, but their costs vary, so
  scheduling and load balancing are worth studying.
- Parallelizing the mode-3 products (`Tmprod`), which currently rely only on
  Eigen and have no explicit parallelism.
- Parallel sorting and prefix sums for Algorithm 2's energy-based rank selection.
- Comparing strategies such as static vs. dynamic scheduling or our own thread
  pool vs. Eigen's internal threading, and producing strong and weak scaling
  plots across thread counts on the 32-core benchmark machine.

Threading must use `<thread>` and related standard headers (see Style.md), not
OpenMP, pthreads, or Win32 threads.

## Benchmarking setup

- Target machine: AMD 7975WX, 32 cores with SMT off, 128 GB DDR5 (8x16),
  Windows 11 Pro. The code should also build and run on Linux.
- Datasets are NetCDF files from the ETH Zürich CGL scientific data collection.
  See `data/Data Sources.txt`. Data files (`*.nc`, `*.mat`) are gitignored.
- Time each configuration once. Algorithm 1 measures uniform truncation
  ranks 1, half the maximum (rounded down), and the maximum. Algorithm 2
  measures `gamma in {0.9, 0.99, 0.999, 0.9999}`.
- Baseline: the existing serial and `parfor` MATLAB implementations.

When porting, match MATLAB's semantics, including column-major layout and
1-based modes in the public API, so results can be checked against MATLAB
output.

## Code layout

- `src/tensor.*`: the `Tensor` type, a dense column-major `Eigen::VectorXd` with
  dimensions.
- `src/permute.*`, `src/tmprod.*`, and `src/mode3_tensor_matrix_product.hpp`:
  ports of MATLAB `permute`, Tensorlab `tmprod`, and `mode3TensorMatrixProduct`.
- `src/main.cpp`: a temporary test harness until the real algorithms exist.
- Dependencies are Eigen (SVD and dense linear algebra) and netcdf-cxx4 (I/O),
  fetched and built by the CMake superbuild in `cmake/`.

## Building

The project builds on Windows through `just` and `scripts/run-cmake.ps1` (see
README.md): `just run`, `just build`, `just debug`, and `just fetch-deps`. CMake
presets are in `CMakePresets.json`.

## Coding rules

Follow **Style.md** strictly. The main points:

- Every function, struct, and enum gets a full doc comment, covering every
  parameter, return value, and exception, with `FIELDS:` and `MEMBERS:` lists.
- No `new` or `delete`. Prefer fixed-width integer types.
- Always use braces. Fail loudly instead of silently succeeding.
- Earn complexity: no templates or attributes until a real need appears.
- Format with `git clang-format`.
