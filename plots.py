"""Figures for the report (static PNGs, light theme)."""
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# fixed categorical order: colour follows the method, never its rank
METHOD_COLOR = {
    "QAOA (penalty)": "#2a78d6",
    "CVaR-QAOA (penalty, a=0.1)": "#eb6834",
    "XY-QAOA": "#1baf7a",
    "XY-CVaR-QAOA (a=0.1)": "#6250d6",
}
MARKERS = {"QAOA (penalty)": "o", "CVaR-QAOA (penalty, a=0.1)": "s", "XY-QAOA": "^", "XY-CVaR-QAOA (a=0.1)": "D"}
BLUE_SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=160)
    plt.close(fig)


def fig_problem(out):
    from drugcombo.data import S, SHORT, PATHOGENS, Instance
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_SEQ)
    inst = Instance()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    im = ax.imshow(S, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(PATHOGENS)), PATHOGENS, rotation=40, ha="right")
    ax.set_yticks(range(len(SHORT)), SHORT)
    ax.grid(False)
    ax.set_title("Susceptibility S[drug, pathogen] (synthetic)")
    fig.colorbar(im, ax=ax, fraction=0.04, label="fraction of isolates covered")
    ax = axes[1]
    M = inst.beta * inst.T + inst.delta * inst.D
    im = ax.imshow(M, cmap=cmap, vmin=0, aspect="auto")
    ax.set_xticks(range(len(SHORT)), SHORT, rotation=40, ha="right")
    ax.set_yticks(range(len(SHORT)), SHORT)
    ax.grid(False)
    ax.set_title("Pairwise penalty: interactions + class duplication")
    fig.colorbar(im, ax=ax, fraction=0.04, label="QUBO coupling added")
    _save(fig, out, "fig_problem.png")


def _agg(runs, key):
    d = defaultdict(list)
    for r in runs:
        d[(r["method"], r["reps"])].append(r["metrics"][key])
    return d


def fig_benchmark(results, out):
    runs = [r for r in results["runs"] if r["tag"] == "main"]
    rnd = results["classical"]["icu_sepsis"]["random"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [("p_optimal", "P(measure the optimal regimen)", True),
              ("p_top5", "P(measure a top-5 regimen)", False),
              ("p_feasible", "P(measure a feasible regimen)", False)]
    for ax, (key, title, logy) in zip(axes, panels):
        agg = _agg(runs, key)
        for meth, color in METHOD_COLOR.items():
            ps = sorted(p for (m, p) in agg if m == meth)
            if not ps:
                continue
            mean = np.array([np.mean(agg[(meth, p)]) for p in ps])
            std = np.array([np.std(agg[(meth, p)]) for p in ps])
            ax.errorbar(ps, mean, yerr=std, color=color, marker=MARKERS[meth], ms=8, lw=2,
                        capsize=3, label=meth, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if key == "p_optimal":
            ax.axhline(rnd["p_opt_uniform_weight_k"], color=INK2, ls="--", lw=1)
            ax.text(1, rnd["p_opt_uniform_weight_k"] * 1.12, "random 3-drug guess (1/120)", color=INK2, fontsize=8)
            ax.axhline(rnd["p_opt_uniform_all"], color=INK2, ls=":", lw=1)
            ax.text(1, rnd["p_opt_uniform_all"] * 1.12, "random bitstring (1/1024)", color=INK2, fontsize=8)
        if key == "p_top5":
            ax.axhline(5 / 120, color=INK2, ls="--", lw=1)
            ax.text(1, 5 / 120 * 1.04, "random 3-drug guess", color=INK2, fontsize=8)
        if key == "p_feasible":
            ax.axhline(rnd["p_feas_uniform_all"], color=INK2, ls=":", lw=1)
            ax.text(1, rnd["p_feas_uniform_all"] + 0.02, "random bitstring", color=INK2, fontsize=8)
            ax.set_ylim(0, 1.05)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("QAOA depth p")
        ax.set_xticks(sorted({p for (_, p) in agg}))
        ax.set_title(title)
    axes[0].legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, -0.18), ncol=2)
    _save(fig, out, "fig_benchmark.png")


def fig_penalty(results, out):
    runs = [r for r in results["runs"] if r["tag"] == "penalty_sweep"]
    if not runs:
        return
    d = defaultdict(list)
    for r in runs:
        d[r["kw"]["penalty"]].append(r["metrics"])
    Ps = sorted(d)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in [(axes[0], "p_feasible", "P(feasible) vs penalty weight"),
                           (axes[1], "p_optimal", "P(optimal) vs penalty weight")]:
        mean = [np.mean([m[key] for m in d[P]]) for P in Ps]
        std = [np.std([m[key] for m in d[P]]) for P in Ps]
        ax.errorbar(Ps, mean, yerr=std, color=METHOD_COLOR["QAOA (penalty)"], marker="o", ms=8, lw=2,
                    capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("penalty weight P (log scale)")
        ax.set_title(title)
    _save(fig, out, "fig_penalty.png")


def fig_distribution(results, out):
    from drugcombo.data import Instance
    from drugcombo.formulation import brute_force, regimen, is_feasible
    from drugcombo.solvers import decode
    inst = Instance()
    ref = brute_force(inst)
    x_opt = tuple(ref[0][2])
    runs = [r for r in results["runs"] if r["tag"] == "main"]
    best_xy = max((r for r in runs if r["method"].startswith("XY")), key=lambda r: r["metrics"]["p_optimal"])
    best_pen = max((r for r in runs if not r["method"].startswith("XY")), key=lambda r: r["metrics"]["p_optimal"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, r in zip(axes, [best_pen, best_xy]):
        top = r["top_states"][:12]
        labels, vals, cols = [], [], []
        for key, p in top:
            x = decode(int(key), inst.n)
            if tuple(x) == x_opt:
                c, tag = "#6250d6", "  (optimal)"
            elif is_feasible(inst, x):
                c, tag = "#86b6ef", ""
            else:
                c, tag = "#c3c2b7", "  (infeasible)"
            labels.append(regimen(x) + tag)
            vals.append(p)
            cols.append(c)
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, color=cols, height=0.7, edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlabel("probability")
        ax.set_title(f"{r['method']}, p={r['reps']}: most likely outcomes")
        ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_distribution.png")


def fig_convergence(results, out):
    runs = [r for r in results["runs"] if r["tag"] == "main" and r["seed"] == 0]
    if not runs:
        return
    pmax = max(r["reps"] for r in runs)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for r in runs:
        if r["reps"] != pmax:
            continue
        h = np.array(r["history"])
        h = (h - h.min()) / (h.max() - h.min() + 1e-12)
        ax.plot(h, color=METHOD_COLOR.get(r["method"], INK2), lw=2, label=r["method"])
    ax.set_xlabel("optimizer evaluation (best start)")
    ax.set_ylabel("objective (min-max scaled)")
    ax.set_title(f"COBYLA convergence, p={pmax}, seed 0")
    ax.legend(fontsize=8)
    _save(fig, out, "fig_convergence.png")


def make_all(results, out):
    fig_problem(out)
    fig_benchmark(results, out)
    fig_penalty(results, out)
    fig_distribution(results, out)
    fig_convergence(results, out)
