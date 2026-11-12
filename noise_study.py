"""
Noise study: how do the optimised circuits behave on a noisy, limited-connectivity device?

    pip install qiskit-aer
    python noise_study.py            # needs results/results.json from run_experiments.py

For each method (seed-0 optimised angles, depth p) we
  1. route the circuit onto an IBM heavy-hex coupling map (or all-to-all),
  2. simulate with Qiskit Aer under depolarising gate noise + readout error,
     with error rates scaled by lambda in {0, 0.1, 0.25, 0.5, 1} x "IBM-like" base rates,
  3. report raw P(optimal) / P(feasible), and P(optimal) after post-selection:
       * symmetry verification (XY-QAOA only): a correct circuit ALWAYS outputs exactly k
         drugs, so any shot with a different count is a detected hardware error -> discard;
       * feasibility filter (any method): discard shots violating either constraint.

Windows note: qiskit-aer fails to load ("filename too long") if the venv path is very
long; put the venv in a short path such as C:\\venvs\\quantarx.
"""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from drugcombo.data import Instance
from drugcombo.formulation import to_qubo, brute_force, is_feasible, regimen
from drugcombo.solvers import normalise, xy_hamiltonian, xy_qaoa_ansatz

OUT = "results"
BASE = {"p1": 2.5e-4, "p2": 7.5e-3, "readout": 1.5e-2}   # roughly IBM Eagle/Heron medians
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0]
SHOTS = 20000
BASIS = ["cx", "rz", "sx", "x"]
METHODS = ["QAOA (penalty)", "XY-QAOA", "XY-CVaR-QAOA (a=0.1)"]


def noise_model(lam: float) -> NoiseModel | None:
    if lam == 0:
        return None
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(lam * BASE["p1"], 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(lam * BASE["p2"], 2), ["cx"])
    r = lam * BASE["readout"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm


def build_circuit(inst, method, reps, point):
    """Rebuild exactly the ansatz the optimiser used and bind its optimal angles."""
    if method.startswith("XY"):
        op, _ = xy_hamiltonian(inst)
        qc = xy_qaoa_ansatz(normalise(op), inst.n, inst.k, reps)
    else:
        _, qubo, _, _ = to_qubo(inst)
        op, _ = qubo.to_ising()
        qc = QAOAAnsatz(normalise(op), reps=reps)
    qc = qc.assign_parameters(point)
    qc.measure_all()
    return qc


def analyse(inst, counts, x_opt):
    tot = sum(counts.values())
    p_opt = p_feas = n_wk = n_wk_opt = 0
    for bits, c in counts.items():
        x = np.array([int(b) for b in bits.replace(" ", "")[::-1]])   # little-endian
        f = is_feasible(inst, x)
        o = bool(np.array_equal(x, x_opt))
        p_opt += c * o
        p_feas += c * f
        if x.sum() == inst.k:
            n_wk += c
            n_wk_opt += c * o
    return {"p_optimal": p_opt / tot, "p_feasible": p_feas / tot,
            "p_opt_given_feasible": p_opt / p_feas if p_feas else 0.0,
            "keep_rate_symmetry": n_wk / tot,
            "p_opt_given_symmetry": n_wk_opt / n_wk if n_wk else 0.0}


def main():
    with open(os.path.join(OUT, "results.json")) as f:
        res = json.load(f)
    inst = Instance()
    ref = brute_force(inst)
    x_opt = ref[0][2]
    print("optimum:", regimen(x_opt))
    pmax = max(r["reps"] for r in res["runs"] if r["tag"] == "main")
    runs = {r["method"]: r for r in res["runs"]
            if r["tag"] == "main" and r["seed"] == 0 and r["reps"] == pmax}

    topologies = {"all-to-all": None, "heavy-hex": CouplingMap.from_heavy_hex(3)}
    out = {"base_rates": BASE, "shots": SHOTS, "reps": pmax, "rows": []}
    for method in METHODS:
        r = runs[method]
        qc = build_circuit(inst, method, pmax, r["optimal_point"])
        for topo, cmap in topologies.items():
            tqc = transpile(qc, basis_gates=BASIS, coupling_map=cmap,
                            optimization_level=3, seed_transpiler=7)
            ops = tqc.count_ops()
            for lam in LAMBDAS:
                t = time.time()
                sim = AerSimulator(noise_model=noise_model(lam), seed_simulator=11)
                counts = sim.run(tqc, shots=SHOTS).result().get_counts()
                m = analyse(inst, counts, x_opt)
                row = {"method": method, "topology": topo, "lambda": lam,
                       "cx": int(ops.get("cx", 0)), "depth": tqc.depth(),
                       "ideal_p_optimal": r["metrics"]["p_optimal"], **m}
                out["rows"].append(row)
                print(f"{method:22s} {topo:10s} lam={lam:<5} CX={row['cx']:4d} "
                      f"P(opt)={m['p_optimal']:.4f} P(feas)={m['p_feasible']:.3f} "
                      f"P(opt|feas)={m['p_opt_given_feasible']:.4f} "
                      f"keep(sym)={m['keep_rate_symmetry']:.3f} ({time.time() - t:.0f}s)", flush=True)

    with open(os.path.join(OUT, "noise_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots import METHOD_COLOR, MARKERS, SURFACE, INK2, _save
    rows = out["rows"]
    rnd_k, rnd_feas = 1 / 120, 1 / 112
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [("p_optimal", "P(optimal), raw shots", rnd_k, "random 3-drug guess"),
              ("p_feasible", "P(feasible), raw shots", None, None),
              ("p_opt_given_feasible", "P(optimal | feasible), after filtering", rnd_feas,
               "random feasible guess")]
    for ax, (key, title, base, blabel) in zip(axes, panels):
        for method in METHODS:
            for topo, ls in [("all-to-all", "-"), ("heavy-hex", "--")]:
                rs = sorted((r for r in rows if r["method"] == method and r["topology"] == topo),
                            key=lambda r: r["lambda"])
                ax.plot([r["lambda"] for r in rs], [r[key] for r in rs], ls=ls, lw=2,
                        marker=MARKERS[method], ms=7, color=METHOD_COLOR[method],
                        markeredgecolor=SURFACE, markeredgewidth=1.2,
                        label=f"{method} · {topo} ({rs[0]['cx']} CX)")
        if base:
            ax.axhline(base, color=INK2, ls=":", lw=1)
            ax.text(0.0, base * 1.06, blabel, color=INK2, fontsize=8)
        ax.set_xlabel("noise scale λ (1 = IBM-like base rates)")
        ax.set_title(title)
        if key == "p_feasible":
            ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)
    _save(fig, OUT, "fig_noise.png")


if __name__ == "__main__":
    main()
