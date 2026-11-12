"""
Quantum and classical solvers.

Quantum (all run on Qiskit's exact StatevectorSampler, no hardware needed):
  * qaoa_penalty  - standard QAOA, X mixer, both constraints as QUBO penalties
  * qaoa_penalty with cvar < 1 - CVaR-QAOA (optimise the best alpha-fraction of shots)
  * qaoa_xy       - constraint-preserving QAOA: Dicke-state start + ring XY mixer keeps
                    exactly k drugs selected, so the cardinality penalty disappears

Classical references:
  * brute force (exact), NumPyMinimumEigensolver (exact diagonalisation of the QUBO),
    greedy heuristic, uniform random sampling.
"""
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, XXPlusYYGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.minimum_eigensolvers import QAOA, SamplingVQE, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA

from .data import Instance
from .dicke import dicke_state
from .formulation import (build_program, to_qubo, surrogate_objective, is_feasible,
                          brute_force, qubo_coefficients)


# ---------------------------------------------------------------- results ------

@dataclass
class RunResult:
    method: str
    reps: int
    seed: int
    probs: dict                     # bitstring-int -> exact probability at optimal params
    metrics: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    num_qubits: int = 0
    depth: int = 0
    cx: int = 0
    optimal_point: np.ndarray | None = None


def decode(key: int, n: int) -> np.ndarray:
    """Qiskit little-endian: bit i of the integer = qubit i = variable i."""
    return np.array([(key >> i) & 1 for i in range(n)])


def evaluate_distribution(inst: Instance, probs: dict, ref) -> dict:
    """Solution-quality metrics of a probability distribution over assignments."""
    feas_vals = [r[0] for r in ref]
    f_min, f_max = min(feas_vals), max(feas_vals)
    x_opt = ref[0][2]
    top5 = {tuple(r[2]) for r in ref[:5]}
    p_top5 = 0.0
    p_feas = p_opt = exp_ratio_num = 0.0
    best = None
    for key, p in probs.items():
        if p < 1e-12:
            continue
        x = decode(key, inst.n)
        if is_feasible(inst, x):
            f = surrogate_objective(inst, x)
            p_feas += p
            exp_ratio_num += p * (f_max - f) / (f_max - f_min)
            if best is None or f < best[0]:
                best = (f, x)
            if np.array_equal(x, x_opt):
                p_opt += p
            if tuple(x) in top5:
                p_top5 += p
    top_key = max(probs, key=probs.get)
    top_x = decode(top_key, inst.n)
    return {
        "p_feasible": p_feas,
        "p_optimal": p_opt,
        "p_top5": p_top5,
        "approx_ratio_given_feasible": exp_ratio_num / p_feas if p_feas else 0.0,
        "most_likely_is_optimal": bool(np.array_equal(top_x, x_opt)),
        "most_likely": top_x.tolist(),
        "most_likely_prob": probs[top_key],
        "best_sampled_value": best[0] if best else None,
        "optimal_value": f_min,
    }


def _exact_probs(circuit: QuantumCircuit, params) -> dict:
    qc = circuit.remove_final_measurements(inplace=False)
    sv = Statevector(qc.assign_parameters(params))
    pr = np.abs(sv.data) ** 2
    return {i: float(p) for i, p in enumerate(pr) if p > 1e-12}


def _ramp(reps: int, rng) -> np.ndarray:
    """Linear-ramp (TQA-style) initial point with a random total time,
    ordered [betas..., gammas...] like Qiskit's sorted parameter order."""
    dt = rng.uniform(0.2, 2.0)
    t = (np.arange(reps) + 0.5) / reps
    return np.concatenate([(1 - t) * dt, t * dt]) + rng.normal(0, 0.1, 2 * reps)


def normalise(op):
    """Scale H so its energy standard deviation over uniform bitstrings is 1.
    For a Z-only Ising operator that std is sqrt(sum c_i^2): computable from the
    coefficients alone, so this scales to large n (no 2^n enumeration needed)."""
    return op / np.sqrt(np.sum(np.abs(op.coeffs) ** 2))


def _circuit_stats(circuit: QuantumCircuit):
    from qiskit import transpile
    qc = circuit.remove_final_measurements(inplace=False)
    qc = qc.assign_parameters(np.full(qc.num_parameters, 0.37))
    t = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    return t.depth(), t.count_ops().get("cx", 0)


def _pass_manager():
    """Transpile each ansatz once to a simple basis (avoids re-synthesising the
    multi-controlled Dicke gates on every sampler call)."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    return generate_preset_pass_manager(optimization_level=1,
                                        basis_gates=["cx", "rz", "ry", "rx", "sx", "x", "h"],
                                        seed_transpiler=0)


def interp(point: np.ndarray) -> np.ndarray:
    """INTERP warm start (Zhou et al., PRX 2020): optimal depth-p angles ->
    initial depth-(p+1) angles by linear interpolation. Order [betas..., gammas...]."""
    p = len(point) // 2
    out = []
    for v in (point[:p], point[p:]):
        ext = np.concatenate([[0.0], v, [0.0]])
        out.append(np.array([(i / p) * ext[i] + ((p - i) / p) * ext[i + 1] for i in range(p + 1)]))
    return np.concatenate(out)


def _multistart(make_algo, op, reps, seed, starts, warm_start=None):
    """Run a variational solver from several initial points, keep the best objective.
    If warm_start is given it is used as the first initial point."""
    rng = np.random.default_rng(seed)
    best, best_hist = None, None
    for s in range(starts):
        hist = []
        x0 = warm_start if (warm_start is not None and s == 0) else _ramp(reps, rng)
        algo = make_algo(x0, hist, seed * 100 + s)
        res = algo.compute_minimum_eigenvalue(op)
        if best is None or res.optimal_value < best.optimal_value:
            best, best_hist = res, hist
    return best, best_hist


# ---------------------------------------------------------------- quantum ------

def qaoa_penalty(inst: Instance, reps=2, seed=0, penalty=None, cvar=None,
                 maxiter=400, shots=16384, starts=4, warm_start=None, ref=None) -> RunResult:
    """Standard QAOA on the fully penalised QUBO (X mixer)."""
    _, qubo, _, P = to_qubo(inst, penalty)
    op, _ = qubo.to_ising()
    op = normalise(op)

    def make(x0, hist, s):
        return QAOA(StatevectorSampler(default_shots=shots, seed=s), COBYLA(maxiter=maxiter, tol=1e-6),
                    reps=reps, initial_point=x0, aggregation=cvar, pass_manager=_pass_manager(),
                    callback=lambda n, x, v, m: hist.append(float(np.real(v))))

    res, hist = _multistart(make, op, reps, seed, starts, warm_start)
    probs = _exact_probs(res.optimal_circuit, res.optimal_point)
    name = "QAOA (penalty)" if cvar is None else f"CVaR-QAOA (penalty, a={cvar})"
    rr = RunResult(name, reps, seed, probs, history=hist, num_qubits=op.num_qubits)
    rr.metrics = evaluate_distribution(inst, probs, ref or brute_force(inst))
    rr.metrics["penalty"] = P
    rr.depth, rr.cx = _circuit_stats(res.optimal_circuit)
    rr.optimal_point = np.asarray(res.optimal_point)
    return rr


def xy_qaoa_ansatz(cost_op, n: int, k: int, reps: int) -> QuantumCircuit:
    """Constraint-preserving QAOA: |Dicke(n,k)> then [cost, ring-XY mixer] x reps."""
    beta = ParameterVector("beta", reps)
    gamma = ParameterVector("gamma", reps)
    qc = QuantumCircuit(n)
    qc.compose(dicke_state(n, k), inplace=True)      # O(n*k)-gate Dicke preparation
    for layer in range(reps):
        qc.append(PauliEvolutionGate(cost_op, time=gamma[layer]), range(n))
        # ring XY mixer in two commuting "brick" layers; each XX+YY swap
        # moves an excitation between neighbours, so Hamming weight (=k) is conserved
        for start in (0, 1):
            for i in range(start, n, 2):
                j = (i + 1) % n
                if i == j:
                    continue
                qc.append(XXPlusYYGate(2 * beta[layer]), [i, j])
    return qc


def xy_hamiltonian(inst: Instance, nephro_penalty=1.0):
    """Cost operator for XY-QAOA: objective + nephrotoxicity penalty only
    (the regimen-size constraint is enforced by the circuit, not by a penalty)."""
    qp = build_program(inst)
    qp.remove_linear_constraint("regimen_size")
    qubo = QuadraticProgramToQubo(penalty=nephro_penalty).convert(qp)  # VAN+GEN<=1 -> P*x_VAN*x_GEN
    assert qubo.get_num_vars() == inst.n, "no slack qubits expected"
    op, _ = qubo.to_ising()
    return op, qubo


def qaoa_xy(inst: Instance, reps=2, seed=0, maxiter=400, shots=16384, cvar=None,
            starts=4, nephro_penalty=1.0, warm_start=None, ref=None) -> RunResult:
    """XY-mixer QAOA: cardinality enforced by the circuit, only nephrotoxicity penalised."""
    op, _ = xy_hamiltonian(inst, nephro_penalty)
    op = normalise(op)
    ansatz = xy_qaoa_ansatz(op, inst.n, inst.k, reps)
    ansatz_t = _pass_manager().run(ansatz)          # same parameters, simple gates

    def make(x0, hist, s):
        return SamplingVQE(StatevectorSampler(default_shots=shots, seed=s), ansatz_t,
                           COBYLA(maxiter=maxiter, tol=1e-6), initial_point=x0, aggregation=cvar,
                           callback=lambda n, x, v, m: hist.append(float(np.real(v))))

    res, hist = _multistart(make, op, reps, seed, starts, warm_start)
    probs = _exact_probs(ansatz_t, res.optimal_point)
    name = "XY-QAOA" if cvar is None else f"XY-CVaR-QAOA (a={cvar})"
    rr = RunResult(name, reps, seed, probs, history=hist, num_qubits=inst.n)
    rr.metrics = evaluate_distribution(inst, probs, ref or brute_force(inst))
    rr.depth, rr.cx = _circuit_stats(ansatz)
    rr.optimal_point = np.asarray(res.optimal_point)
    return rr


# ---------------------------------------------------------------- classical ----

def exact_eigensolver(inst: Instance, penalty=None):
    """Exact diagonalisation of the penalised QUBO Hamiltonian (validates the penalty)."""
    _, qubo, _, _ = to_qubo(inst, penalty)
    op, offset = qubo.to_ising()
    res = NumPyMinimumEigensolver().compute_minimum_eigenvalue(op)
    key = int(np.argmax(np.abs(res.eigenstate.data)))
    x = decode(key, inst.n)
    return x, float(np.real(res.eigenvalue) + offset)


def greedy(inst: Instance):
    """Add the drug with the best marginal objective change, k times (respecting constraints)."""
    x = np.zeros(inst.n, dtype=int)
    for _ in range(inst.k):
        best = None
        for d in np.flatnonzero(x == 0):
            y = x.copy(); y[d] = 1
            if y[inst.nephro_idx].sum() > 1:
                continue
            f = surrogate_objective(inst, y)
            if best is None or f < best[0]:
                best = (f, d)
        x[best[1]] = 1
    return x, surrogate_objective(inst, x)


def random_baseline(inst: Instance, ref):
    """Uniform random guessing over all 2^n bitstrings and over all weight-k strings."""
    n_all = 2 ** inst.n
    n_k = sum(1 for _ in combinations(range(inst.n), inst.k))
    return {"p_opt_uniform_all": 1 / n_all, "p_feas_uniform_all": len(ref) / n_all,
            "p_opt_uniform_weight_k": 1 / n_k, "p_feas_uniform_weight_k": len(ref) / n_k}
