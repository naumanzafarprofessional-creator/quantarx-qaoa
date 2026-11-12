"""
Reproduce every number and figure in the README.

    python run_experiments.py           # full benchmark (~15-20 min on 8 cores)
    python run_experiments.py --quick   # 2-minute demo (1 seed, p = 1..2)

Outputs go to results/: results.json, summary.md and the figures.
"""
import argparse, json, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np

warnings.filterwarnings("ignore")
OUT = "results"


# ----------------------------------------------------------------- workers -----

def _job(args):
    """Run one quantum experiment for depths p = 1..pmax with INTERP warm starts
    (in a worker process). Returns one record per depth."""
    import warnings; warnings.filterwarnings("ignore")
    from drugcombo.data import Instance
    from drugcombo.formulation import brute_force
    from drugcombo.solvers import qaoa_penalty, qaoa_xy, interp
    tag, method, pmax, seed, kw, scenario, starts = args
    inst = Instance(scenario=scenario)
    ref = brute_force(inst)
    solver = qaoa_penalty if method == "penalty" else qaoa_xy
    out, warm = [], None
    for reps in range(1, pmax + 1):
        t = time.time()
        # depth 1: multi-start; deeper: INTERP warm start + one fresh random start
        r = solver(inst, reps=reps, seed=seed, starts=starts if reps == 1 else 2,
                   warm_start=warm, ref=ref, **kw)
        warm = interp(r.optimal_point)
        top = sorted(r.probs.items(), key=lambda kv: -kv[1])[:20]
        out.append({"tag": tag, "method": r.method, "reps": reps, "seed": seed, "scenario": scenario,
                    "kw": kw, "metrics": r.metrics, "history": r.history, "depth": r.depth,
                    "cx": r.cx, "top_states": top, "seconds": time.time() - t,
                    "optimal_point": r.optimal_point.tolist()})
    return out


# ----------------------------------------------------------------- main --------

def main(quick: bool):
    from drugcombo.data import Instance, SHORT, PATHOGENS
    from drugcombo.formulation import (brute_force, regimen, true_coverage, default_penalty,
                                       to_qubo)
    from drugcombo.solvers import exact_eigensolver, greedy, random_baseline

    os.makedirs(OUT, exist_ok=True)
    results = {"classical": {}, "runs": []}

    # ---- E0: classical / exact references ------------------------------------
    for sc in ["icu_sepsis", "community_pneumonia"]:
        inst = Instance(scenario=sc)
        ref = brute_force(inst)
        x_eig, e_eig = exact_eigensolver(inst)
        x_gr, f_gr = greedy(inst)
        true_best = min(ref, key=lambda r: r[1])
        _, qubo, _, P = to_qubo(inst)
        results["classical"][sc] = {
            "num_qubits": qubo.get_num_vars(),
            "penalty_default": P,
            "num_feasible": len(ref),
            "top10": [{"regimen": regimen(x), "qubo_objective": s, "true_objective": t,
                       "true_coverage": true_coverage(inst, x)} for s, t, x in ref[:10]],
            "worst_feasible_objective": ref[-1][0],
            "eigensolver": {"regimen": regimen(x_eig), "energy": e_eig},
            "greedy": {"regimen": regimen(x_gr), "objective": f_gr},
            "true_objective_optimum": regimen(true_best[2]),
            "random": random_baseline(inst, ref),
        }
        print(f"[{sc}] optimum {regimen(ref[0][2])}  eigensolver {regimen(x_eig)}  "
              f"greedy {regimen(x_gr)}  true-objective optimum {regimen(true_best[2])}")

    # ---- E4 (classical): fidelity of the 2nd-order coverage approximation ----
    sweep = []
    for a in np.round(np.linspace(0.5, 3.0, 11), 2):
        inst = Instance(alpha=float(a))
        ref = brute_force(inst)
        tb = min(ref, key=lambda r: r[1])
        sweep.append({"alpha": float(a), "qubo_opt": regimen(ref[0][2]), "true_opt": regimen(tb[2]),
                      "agree": bool(np.array_equal(ref[0][2], tb[2])),
                      "true_gap": float(ref[0][1] - tb[1])})
    results["alpha_sweep"] = sweep

    # ---- quantum experiments --------------------------------------------------
    seeds = [0] if quick else [0, 1, 2]
    pmax = 2 if quick else 3
    starts = 2 if quick else 4
    inst0 = Instance()
    jobs = []
    # E1: main benchmark (each job sweeps p = 1..pmax)
    for seed in seeds:
        jobs += [("main", "penalty", pmax, seed, {}, "icu_sepsis", starts),
                 ("main", "penalty", pmax, seed, {"cvar": 0.1}, "icu_sepsis", starts),
                 ("main", "xy", pmax, seed, {}, "icu_sepsis", starts),
                 ("main", "xy", pmax, seed, {"cvar": 0.1}, "icu_sepsis", starts)]
    # E2: penalty-weight sensitivity (standard QAOA, p = 2)
    Pdef = default_penalty(inst0)
    for P in ([0.5, Pdef] if quick else [0.25, 0.5, 1.0, 2.0, 4.0, Pdef]):
        for seed in seeds[:2]:
            jobs.append(("penalty_sweep", "penalty", 2, seed, {"penalty": float(P)}, "icu_sepsis", starts))
    # E3: second clinical scenario with the best method
    jobs.append(("scenario", "xy", pmax, 0, {"cvar": 0.1}, "community_pneumonia", starts))

    print(f"running {len(jobs)} quantum jobs on {os.cpu_count()} cores ...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for i, recs in enumerate(ex.map(_job, jobs), 1):
            if recs[0]["tag"] in ("penalty_sweep",):
                recs = [r for r in recs if r["reps"] == 2]      # sweep reports p = 2 only
            if recs[0]["tag"] == "scenario":
                recs = recs[-1:]                                # deepest circuit only
            for res in recs:
                m = res["metrics"]
                print(f"  [{i:3d}/{len(jobs)}] {res['tag']:13s} {res['method']:28s} p={res['reps']} "
                      f"seed={res['seed']} kw={res['kw']}  P(opt)={m['p_optimal']:.4f} "
                      f"P(top5)={m['p_top5']:.3f} P(feas)={m['p_feasible']:.3f}  ({res['seconds']:.0f}s)",
                      flush=True)
                results["runs"].append(res)
    print(f"quantum jobs done in {time.time() - t0:.0f}s")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    from plots import make_all
    make_all(results, OUT)
    write_summary(results)
    print("wrote", OUT)


def write_summary(results):
    from collections import defaultdict
    cl = results["classical"]["icu_sepsis"]
    rnd = cl["random"]
    lines = ["# Results summary (auto-generated)\n",
             f"Qubits: {cl['num_qubits']} | feasible regimens: {cl['num_feasible']} / 1024 | "
             f"default penalty: {cl['penalty_default']}\n",
             "## Classical references (ICU sepsis)\n",
             "| Rank | Regimen | QUBO objective | True objective | True coverage |",
             "|---|---|---|---|---|"]
    for i, r in enumerate(cl["top10"], 1):
        lines.append(f"| {i} | {r['regimen']} | {r['qubo_objective']:.4f} | {r['true_objective']:.4f} | {r['true_coverage']:.3f} |")
    lines += ["", f"- Exact eigensolver ground state: **{cl['eigensolver']['regimen']}**",
              f"- Greedy heuristic: **{cl['greedy']['regimen']}** (objective {cl['greedy']['objective']:.4f})",
              f"- Optimum of the exact non-linear objective: **{cl['true_objective_optimum']}**", ""]

    agg = defaultdict(list)
    for r in results["runs"]:
        if r["tag"] == "main":
            agg[(r["method"], r["reps"])].append(r)
    lines += ["## Quantum benchmark (mean ± std over seeds, exact probabilities at optimised angles)\n",
              "| Method | p | P(optimal) | ×random(weight-3) | P(top-5) | P(feasible) | approx. ratio | CX | depth |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (meth, reps), rs in sorted(agg.items()):
        po = np.array([r["metrics"]["p_optimal"] for r in rs])
        pt = np.array([r["metrics"]["p_top5"] for r in rs])
        pf = np.array([r["metrics"]["p_feasible"] for r in rs])
        ar = np.array([r["metrics"]["approx_ratio_given_feasible"] for r in rs])
        lines.append(f"| {meth} | {reps} | {po.mean():.4f} ± {po.std():.4f} | {po.mean() / rnd['p_opt_uniform_weight_k']:.1f}× "
                     f"| {pt.mean():.3f} | {pf.mean():.3f} | {ar.mean():.3f} | {rs[0]['cx']} | {rs[0]['depth']} |")
    lines += ["", f"Random guessing: P(opt) = {rnd['p_opt_uniform_all']:.4f} over all bitstrings, "
              f"{rnd['p_opt_uniform_weight_k']:.4f} over 3-drug regimens; "
              f"P(feasible) = {rnd['p_feas_uniform_all']:.3f} over all bitstrings.", ""]

    pen = defaultdict(list)
    for r in results["runs"]:
        if r["tag"] == "penalty_sweep":
            pen[r["kw"]["penalty"]].append(r)
    lines += ["## Penalty sensitivity (standard QAOA, p = 2)\n", "| Penalty | P(optimal) | P(feasible) |", "|---|---|---|"]
    for P, rs in sorted(pen.items()):
        lines.append(f"| {P} | {np.mean([r['metrics']['p_optimal'] for r in rs]):.4f} | "
                     f"{np.mean([r['metrics']['p_feasible'] for r in rs]):.3f} |")

    lines += ["", "## Coverage-approximation fidelity (alpha sweep)\n", "| alpha | QUBO optimum | true optimum | agree |", "|---|---|---|---|"]
    for s in results["alpha_sweep"]:
        lines.append(f"| {s['alpha']} | {s['qubo_opt']} | {s['true_opt']} | {'yes' if s['agree'] else 'no'} |")

    sc = [r for r in results["runs"] if r["tag"] == "scenario"]
    if sc:
        c2 = results["classical"]["community_pneumonia"]
        m = sc[0]["metrics"]
        lines += ["", "## Second scenario: community-acquired pneumonia\n",
                  f"- Exact optimum: **{c2['top10'][0]['regimen']}**",
                  f"- {sc[0]['method']} p={sc[0]['reps']}: P(optimal) = {m['p_optimal']:.4f}, P(top-5) = {m['p_top5']:.3f}, "
                  f"P(feasible) = {m['p_feasible']:.3f}"]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast demo run")
    main(ap.parse_args().quick)
