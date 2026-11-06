#!/usr/bin/env python3
"""Figures and tables of the report, computed only from results/raw/*.csv.

    python3 analysis/plot_results.py --raw results/raw --out results/report_assets

No measurement is typed into this script.  Before drawing anything it checks that
every run on the same data built the SAME tree (same hash), so a plot can never
mix runs of different algorithms or of a broken parallel version.

Definitions
    speedup(p)     = T_serial / T(p)            T_serial: the dt_serial build, same data
    efficiency(p)  = speedup(p) / p              p = ranks x threads
    weak eff.(p)   = T(p_min) / T(p)             work per core held constant
    t_other        = t_total - (hist + scan + part + wait + allreduce)
Error bars are min..max over all repetitions of a configuration.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

PHASES = ["t_hist", "t_scan", "t_part", "t_wait", "t_allreduce", "t_other"]
PHASE_LABEL = {"t_hist": "local histograms", "t_scan": "split search (replicated)", "t_part": "partition / slot update",
               "t_wait": "waiting for slowest rank", "t_allreduce": "MPI_Allreduce transfer", "t_other": "other"}


def plain_log_axes(ax, xvals, y=False):
    """Log axes labelled with the actual core counts instead of 2 x 10^k."""
    ax.set_xticks(sorted(set(int(v) for v in xvals)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if y:
        lo, hi = ax.get_ylim()
        ax.set_yticks([2 ** k for k in range(-4, 12) if lo * 0.99 <= 2 ** k <= hi * 1.01])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def load(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        sys.exit(f"no CSV files in {raw_dir}: run scripts/run_matrix.sh first")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        d = d.drop(columns=[c for c in d.columns if c == "RESULT"])
        d["file"] = os.path.basename(f)
        frames.append(d)
    if not frames:
        sys.exit("CSV files contain no rows")
    df = pd.concat(frames, ignore_index=True)
    df["t_other"] = (df["t_total"] - df[["t_hist", "t_scan", "t_part", "t_wait", "t_allreduce"]].sum(axis=1)).clip(lower=0)
    print(f"[plot] {len(df)} rows from {len(files)} files:")
    for f in files:
        print("        ", os.path.basename(f))
    return df


def check_trees(df, allow):
    df = df.copy()
    df["real"] = df["tag"].str.startswith("real")
    bad = False
    print("[plot] correctness: every configuration on the same data must build the same tree")
    for (real, N, F, depth), g in df.groupby(["real", "N", "F", "depth"]):
        hashes = g["tree_hash"].astype(str).nunique()
        status = "OK " if hashes == 1 else "MISMATCH"
        if hashes != 1:
            bad = True
        print(f"        {status} N={N:>11} F={F:>3} depth={depth}: {hashes} distinct tree(s) over {len(g)} runs, "
              f"{g[['impl', 'ranks', 'threads', 'growth', 'omp']].drop_duplicates().shape[0]} configurations")
    if bad and not allow:
        sys.exit("[plot] refusing to plot: trees differ between configurations (use --allow-mismatch to override)")
    if (df["approx"] != 0).any():
        print("[plot] NOTE: some runs used approximate (quantile) bins; trees are still identical across configurations")


def times(df, tags_pred):
    d = df[(df.profile == 0) & df.tag.map(tags_pred)]
    g = d.groupby(["tag", "ranks", "threads", "cores", "N", "F"])["t_total"].agg(["median", "min", "max", "count"]).reset_index()
    return g.rename(columns={"median": "t", "min": "tmin", "max": "tmax", "count": "n"})


def label_of(tag):
    if tag.endswith("_mpi"):
        return "MPI only (1 thread / rank)"
    if tag.endswith("_thr"):
        return "OpenMP only (1 rank)"
    if "_hyb" in tag:
        return f"MPI x {tag.split('_hyb')[1]} threads / rank"
    return tag


def serial_time(df, prefix, N, F):
    s = df[(df.tag == f"{prefix}_serial") & (df.N == N) & (df.F == F) & (df.profile == 0)]
    if len(s):
        return float(s.t_total.median()), "dt_serial build"
    m = df[(df.tag == f"{prefix}_mpi") & (df.N == N) & (df.F == F) & (df.cores == 1) & (df.profile == 0)]
    if len(m):
        return float(m.t_total.median()), "1 rank x 1 thread of the MPI build (no serial baseline found)"
    return None, ""


def strong_block(df, prefix, out, tables):
    t = times(df, lambda x: x.startswith(prefix + "_") and not x.endswith("_serial"))
    if t.empty:
        print(f"[plot] {prefix}: no rows, skipped")
        return
    N, F = int(t.N.iloc[0]), int(t.F.iloc[0])
    ts, how = serial_time(df, prefix, N, F)
    if ts is None:
        print(f"[plot] {prefix}: no baseline, skipped")
        return
    t = t[(t.N == N) & (t.F == F)].copy()
    t["speedup"] = ts / t["t"]
    t["speedup_lo"] = ts / t["tmax"]
    t["speedup_hi"] = ts / t["tmin"]
    t["efficiency"] = t["speedup"] / t["cores"]
    tables[f"{prefix}_scaling"] = t.assign(t_serial=ts).sort_values(["tag", "cores"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for tag, g in t.groupby("tag"):
        g = g.sort_values("cores")
        yerr = [g.speedup - g.speedup_lo, g.speedup_hi - g.speedup]
        ax[0].errorbar(g.cores, g.speedup, yerr=yerr, marker="o", capsize=3, label=label_of(tag))
        ax[1].plot(g.cores, g.efficiency * 100, marker="o", label=label_of(tag))
    cmax = t.cores.max()
    xs = np.array([1, cmax])
    ax[0].plot(xs, xs, "k--", lw=1, label="ideal")
    ax[0].set(xscale="log", yscale="log", xlabel="cores (ranks x threads)", ylabel=f"speedup vs serial ({ts:.3f} s)",
              title=f"Strong scaling, N={N:,}, F={F}")
    ax[1].axhline(100, color="k", ls="--", lw=1)
    ax[1].set(xscale="log", xlabel="cores (ranks x threads)", ylabel="parallel efficiency (%)", title="Efficiency", ylim=(0, 115))
    plain_log_axes(ax[0], t.cores, y=True)
    plain_log_axes(ax[1], t.cores)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    fig.text(0.01, 0.005, f"baseline: {how}.  Data: {', '.join(sorted(df[df.tag.str.startswith(prefix)].file.unique()))}", fontsize=6, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{prefix}_scaling.png"), dpi=200)
    plt.close(fig)

    # time breakdown from the profiled runs
    p = df[(df.profile == 1) & df.tag.str.startswith(prefix + "_") & (df.N == N) & (df.F == F)]
    if len(p):
        tags = sorted(p.tag.unique())
        fig, ax = plt.subplots(1, len(tags), figsize=(6.2 * len(tags), 4.6), squeeze=False)
        brk = []
        for a, tag in zip(ax[0], tags):
            g = p[p.tag == tag].groupby(["ranks", "threads", "cores"])[["t_total"] + PHASES + ["hist_imbalance"]].mean().reset_index().sort_values("cores")
            g["tag"] = tag
            brk.append(g)
            bottom = np.zeros(len(g))
            for ph in PHASES:
                frac = (g[ph] / g["t_total"]).to_numpy() * 100
                a.bar(range(len(g)), frac, bottom=bottom, label=PHASE_LABEL[ph])
                bottom += frac
            a.set_xticks(range(len(g)))
            a.set_xticklabels([f"{c}\n{t_:.3f}s" for c, t_ in zip(g.cores, g.t_total)], fontsize=7)
            a.set(xlabel="cores   (total time of the profiled runs)", ylabel="% of training time", title=label_of(tag), ylim=(0, 100))
        ax[0][0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"Where the time goes (profiled runs, mean over ranks; N={N:,}, F={F})")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{prefix}_breakdown.png"), dpi=200)
        plt.close(fig)
        tables[f"{prefix}_breakdown"] = pd.concat(brk, ignore_index=True)


def weak_block(df, out, tables):
    t = times(df, lambda x: x.startswith("weak_"))
    if t.empty:
        print("[plot] weak: no rows, skipped")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    rows = []
    for tag, g in t.groupby("tag"):
        g = g.sort_values("cores").copy()
        g["efficiency"] = g["t"].iloc[0] / g["t"]
        g["rows_per_core"] = g.N / g.cores
        rows.append(g)
        ax[0].errorbar(g.cores, g.t, yerr=[g.t - g.tmin, g.tmax - g.t], marker="o", capsize=3, label=label_of(tag.replace("weak", "x")))
        ax[1].plot(g.cores, g.efficiency * 100, marker="o", label=label_of(tag.replace("weak", "x")))
    ax[0].set(xscale="log", xlabel="cores", ylabel="training time (s)", title="Weak scaling: rows per core held constant")
    ax[1].axhline(100, color="k", ls="--", lw=1)
    ax[1].set(xscale="log", xlabel="cores", ylabel="weak-scaling efficiency (%)  = T(smallest) / T(p)", ylim=(0, 115))
    plain_log_axes(ax[0], t.cores)
    plain_log_axes(ax[1], t.cores)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "weak_scaling.png"), dpi=200)
    plt.close(fig)
    tables["weak_scaling"] = pd.concat(rows, ignore_index=True)

    p = df[(df.profile == 1) & df.tag.str.startswith("weak_mpi")]
    if len(p):
        g = p.groupby(["ranks", "cores", "N"])[["t_total"] + PHASES].mean().reset_index().sort_values("cores")
        tables["weak_breakdown"] = g


def hybrid_block(df, out, tables):
    t = times(df, lambda x: x == "hybrid_fixed")
    if t.empty:
        print("[plot] hybrid: no rows, skipped")
        return
    t = t.sort_values("threads")
    tables["hybrid_fixed"] = t
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(t))
    ax.bar(x, t.t, yerr=[t.t - t.tmin, t.tmax - t.t], capsize=3, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}x{th}" for r, th in zip(t.ranks, t.threads)])
    ax.set(xlabel="ranks x threads per rank", ylabel="training time (s)",
           title=f"Same {int(t.cores.iloc[0])} cores, different MPI / OpenMP split (N={int(t.N.iloc[0]):,}, F={int(t.F.iloc[0])})")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hybrid_fixed_cores.png"), dpi=200)
    plt.close(fig)


def design_block(df, out, tables):
    d = df[(df.tag == "design") & (df.profile == 0)]
    if d.empty:
        print("[plot] design: no rows, skipped")
        return
    g = d.groupby(["ranks", "threads", "growth", "omp"]).agg(t=("t_total", "median"), tmin=("t_total", "min"), tmax=("t_total", "max"),
                                                              n_allreduce=("n_allreduce", "first"),
                                                              mbytes=("mbytes_allreduce", "first")).reset_index()
    tables["design_choices"] = g
    cfgs = g[["ranks", "threads"]].drop_duplicates().to_numpy().tolist()
    fig, ax = plt.subplots(1, len(cfgs), figsize=(6 * len(cfgs), 4.4), squeeze=False)
    for a, (r, th) in zip(ax[0], cfgs):
        s = g[(g.ranks == r) & (g.threads == th)].sort_values(["growth", "omp"])
        x = np.arange(len(s))
        a.bar(x, s.t, yerr=[s.t - s.tmin, s.tmax - s.t], capsize=3, color=["#4c72b0" if gr == "level" else "#dd8452" for gr in s.growth])
        a.set_xticks(x)
        a.set_xticklabels([f"{gr}\n{o}\n{int(n)} allreduce\n{mb:.2f} MB" for gr, o, n, mb in zip(s.growth, s.omp, s.n_allreduce, s.mbytes)], fontsize=7)
        a.set(ylabel="training time (s)", title=f"{int(r)} ranks x {int(th)} threads: growth x OpenMP strategy")
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "design_choices.png"), dpi=200)
    plt.close(fig)


def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/report_assets")
    ap.add_argument("--allow-mismatch", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.raw)
    check_trees(df, a.allow_mismatch)

    tables = {}
    for prefix in ("strong", "real", "smoke"):
        strong_block(df, prefix, a.out, tables)
    weak_block(df, a.out, tables)
    hybrid_block(df, a.out, tables)
    design_block(df, a.out, tables)

    md = ["# Summary tables (generated by analysis/plot_results.py)\n"]
    for name, t in tables.items():
        t.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
        md.append(f"## {name}\n")
        md.append(md_table(t))
        md.append("")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md))
    print(f"[plot] wrote {len(tables)} tables and figures to {a.out}")
    for f in sorted(os.listdir(a.out)):
        print("        ", f)


if __name__ == "__main__":
    main()
