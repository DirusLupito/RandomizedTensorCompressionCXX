"""Run the two planar flow benchmarks and plot completed timings, including manual runs."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASETS = ("jungtelziemniak2d", "boussinesq", "tangaroa")
# Scaling and scheduler menu entries for the two automatic experiments.
AUTOMATIC = (("measureJungtelziemniak2d", 11, 14), ("measureBoussinesq", 12, 15))
COLUMNS = "dataset language sweep algorithm execution threads batch_size c gamma repeat seconds m p n".split()
BASE_KEY = "dataset language algorithm c gamma m p n".split()


def run_cpp(data, output):
    """Build once, then select the planar experiments in the ordinary menu."""
    subprocess.run(["just", "build"], cwd=ROOT, check=True)
    selections = [str(number) for _, main, batch in AUTOMATIC for number in (main, batch)]
    subprocess.run(["just", "nocompile", str(data)], cwd=ROOT, check=True,
                   input=" ".join(selections) + "\n\n", text=True,
                   env={**os.environ, "RTC_MEASURED_OUTPUT": str(output)})


def run_matlab(data, output, mode, session):
    """Use an existing shared MATLAB session, or collect manually generated CSVs."""
    if mode == "csv":
        return
    # The Windows engine loads additional MATLAB DLLs when it attaches.
    # Use the installation selected on PATH, including its native libraries.
    matlab = shutil.which("matlab")
    if os.name == "nt" and matlab:
        native = Path(matlab).resolve().parent / "win64"
        if native.is_dir():
            os.environ["PATH"] = str(native) + os.pathsep + os.environ["PATH"]
    try:
        import matlab.engine as engine
    except (ImportError, OSError) as exc:
        if mode == "shared" or session:
            raise RuntimeError("Install MATLAB Engine for this Python to use a shared session.") from exc
        print("MATLAB Engine unavailable. Collecting completed MATLAB CSVs.")
        return
    explicit_session = bool(session)
    names = engine.find_matlab()
    if not names and mode == "auto" and not session:
        print("No shared MATLAB session. Collecting completed MATLAB CSVs.")
        return
    if session:
        if session not in names:
            raise RuntimeError(f"MATLAB session {session!r} is not shared. Available: {names}")
    elif len(names) == 1:
        session = names[0]
    else:
        raise RuntimeError(f"Select a shared MATLAB session with --session. Available: {names}")

    # Connecting by name reuses the operator's TeBaC path and pool. Restore
    # the working directory when control returns to the interactive session.
    try:
        eng = engine.connect_matlab(session)
    except engine.EngineError as exc:
        if mode == "shared" or explicit_session:
            raise
        print(f"Could not attach to {session}: {exc}. Collecting completed MATLAB CSVs.")
        return
    previous = eng.pwd()
    try:
        eng.cd(str(HERE), nargout=0)
        for function, _, _ in AUTOMATIC:
            eng.feval(function, str(data), str(output), nargout=0)
    finally:
        eng.cd(previous, nargout=0)


def collect(output):
    """Match each setting to its serial time, averaging historical five-trial CSVs."""
    tables = []
    keys = [column for column in COLUMNS if column not in ("repeat", "seconds")]
    for dataset in DATASETS:
        paths = list(output.glob(f"{dataset}_matlab_*.csv"))
        paths += list(output.glob(f"{dataset}_cpp_*.csv"))
        for path in sorted(paths):
            table = pd.read_csv(path)
            if list(table.columns) != COLUMNS or table.empty or not table.dataset.eq(dataset).all():
                raise ValueError(f"Unexpected timing schema or dataset in {path}")
            repeats = 1 if table["repeat"].eq(1).all() else 5
            if (not table["repeat"].isin(range(1, repeats + 1)).all()
                    or not table.groupby(keys, dropna=False).size().eq(repeats).all()):
                raise ValueError(f"Expected {repeats} timing(s) per configuration in {path}.")
            tables.append(table)
    if not tables:
        raise ValueError(f"No completed timing CSVs found in {output}")
    raw = pd.concat(tables, ignore_index=True)
    if not np.isfinite(raw.seconds).all() or not raw.seconds.gt(0).all():
        raise ValueError("Every elapsed time must be finite and positive.")
    if raw.duplicated(keys + ["repeat"]).any():
        raise ValueError("Duplicate timing for the same configuration and repetition.")
    summary = raw.groupby(keys, dropna=False).seconds.mean().reset_index(name="mean_seconds")
    serial = summary.loc[(summary.sweep == "scaling") & (summary.execution == "serial")
                         & (summary.threads == 1), BASE_KEY + ["mean_seconds"]]
    serial = serial.rename(columns={"mean_seconds": "serial_seconds"})
    summary = summary.merge(serial, on=BASE_KEY, how="left", validate="many_to_one")
    if summary.serial_seconds.isna().any():
        missing = summary.loc[summary.serial_seconds.isna(), BASE_KEY].iloc[0].to_dict()
        raise ValueError(f"Missing matching serial baseline: {missing}. Run the dataset's scaling experiment.")
    summary["speedup"] = summary.serial_seconds / summary.mean_seconds
    summary["efficiency_pct"] = 100 * summary.speedup / summary.threads
    raw.to_csv(output / "timings.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    if "matlab" not in set(raw.language):
        print("No completed MATLAB CSVs found. Figures contain C++ timings only.")
    return summary


METRICS = (("mean_seconds", "Runtime (s)"), ("speedup", "Speedup vs. serial"),
           ("efficiency_pct", "Efficiency (%)"))
LANGUAGE = {"cpp": "C++", "matlab": "MATLAB"}


def finish(fig, axes, title, xlabel, ylabel, folder, stem, ticks=()):
    """Label the axes and save the same simple figure for screen and print."""
    fig.suptitle(title)
    for ax in axes:
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if ticks:
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks, [f"{p:g}" for p in ticks])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{stem}.{ext}", dpi=150)
    plt.close(fig)


def plot(summary, folder):
    """Plot thread scaling, the complete rank sweep, and the 32-thread batch sweep."""
    folder.mkdir(parents=True, exist_ok=True)
    for (dataset, algorithm), data in summary.groupby(["dataset", "algorithm"]):
        scaling = data.loc[data.sweep == "scaling"]
        languages = sorted(scaling.language.unique())
        parameter = "c" if algorithm == "TSVDM" else "gamma"
        values = sorted(scaling[parameter].unique())
        # Also keep the thread-count plot readable for historical full-rank sweeps.
        shown = values
        if parameter == "c" and len(values) > 3:
            shown = sorted({values[0], max(1, values[-1] // 2), values[-1]}.intersection(values))
        for metric, ylabel in METRICS:
            fig, panels = plt.subplots(1, len(languages), squeeze=False, sharey=True,
                                       figsize=(5 * len(languages), 3.8), layout="constrained")
            for ax, language in zip(panels[0], languages):
                subset = scaling.loc[scaling.language == language]
                ax.set_title(LANGUAGE[language])
                for value in shown:
                    series = subset.loc[subset[parameter] == value].sort_values("threads")
                    # Keep the serial baseline at one. An explicitly opened
                    # one-worker MATLAB pool gets a separate marker at one.
                    one_worker = (series.execution == "parallel") & (series.threads == 1)
                    curve = series.loc[~one_worker]
                    line, = ax.plot(curve.threads, curve[metric], "o-", markersize=4,
                                    label=f"{parameter}={value:g}")
                    if one_worker.any():
                        pts = series.loc[one_worker]
                        ax.plot(pts.threads, pts[metric], "x", color=line.get_color(),
                                label="1-worker pool" if value == shown[0] else "_nolegend_")
            prefix = f"{dataset}_{algorithm}"
            finish(fig, panels[0], f"{dataset} · {algorithm}", "Threads / workers", ylabel,
                   folder, f"{prefix}_threads_{metric}", sorted(scaling.threads.unique()))

            if parameter == "c":
                fig, panels = plt.subplots(1, len(languages), squeeze=False, sharey=True,
                                           figsize=(5 * len(languages), 3.8), layout="constrained")
                for ax, language in zip(panels[0], languages):
                    ax.set_title(LANGUAGE[language])
                    for (execution, p), series in scaling.loc[scaling.language == language].groupby(["execution","threads"]):
                        series = series.sort_values("c")
                        ax.plot(series.c, series[metric], label="serial" if execution == "serial" else f"{p} workers")
                finish(fig, panels[0], f"{dataset} · TSVDM", "Uniform slice rank c", ylabel,
                       folder, f"{prefix}_ranks_{metric}")

            batch = data.loc[data.sweep == "batch"].sort_values("batch_size")
            if not batch.empty:
                fig, ax = plt.subplots(figsize=(5.5, 3.8), layout="constrained")
                value = batch[parameter].iloc[0]
                ax.plot(batch.batch_size, batch[metric], "o-", label=f"{parameter}={value:g}")
                finish(fig, [ax], f"{dataset} · {algorithm} · 32 C++ threads",
                       "Slices per batch", ylabel, folder, f"{prefix}_batch_{metric}", sorted(batch.batch_size.unique()))


def main():
    """Run measurements or regenerate plots from previously collected timing files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "measuredExperiments")
    parser.add_argument("--plot-only", action="store_true", help="Read completed CSVs without running measurements.")
    parser.add_argument("--matlab", choices=["auto", "shared", "csv"], default="auto",
                        help="Use an existing shared MATLAB session or previously saved CSVs.")
    parser.add_argument("--session", help="Name of an already shared MATLAB session.")
    args = parser.parse_args()
    if args.session and args.matlab == "csv":
        parser.error("--session requires --matlab auto or shared")
    data, output = args.data_dir.resolve(), args.output.resolve()
    if not args.plot_only:
        output.mkdir(parents=True, exist_ok=True)
        run_cpp(data, output)
        run_matlab(data, output, args.matlab, args.session)
    plot(collect(output), output / "plots")
    print(f"Saved timings.csv, summary.csv and plots to {output}")


if __name__ == "__main__":
    main()
