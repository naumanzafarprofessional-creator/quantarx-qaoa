"""
Mathematical model.

Decision variables: x_d in {0,1}, 1 if drug d is in the regimen.

True (non-linear) expected coverage of pathogen j by a set X of drugs:
    cov_j(X) = 1 - prod_{d in X} (1 - S[d,j])

Second-order inclusion-exclusion turns this into a QUBO-compatible quadratic form:
    cov_j(x) ~= sum_d S[d,j] x_d  -  sum_{d<e} S[d,j] S[e,j] x_d x_e
The quadratic term is exactly the *redundancy* of two drugs covering the same bug.

Objective (minimise):
    f(x) = - alpha * sum_j w_j [ sum_d S_dj x_d - sum_{d<e} S_dj S_ej x_d x_e ]   (coverage)
           + beta  * sum_{d<e} T_de x_d x_e                                        (interactions)
           + delta * sum_{d<e} D_de x_d x_e                                        (class duplication)
           + gamma * sum_d c_d x_d                                                 (stewardship)
Constraints:
    sum_d x_d == k                         (regimen size)
    sum_{d in nephrotoxic} x_d <= 1        (at most one nephrotoxic drug)
"""
from itertools import combinations, product
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from .data import Instance, SHORT


def qubo_coefficients(inst: Instance):
    """Return (linear[n], quadratic[n,n] upper-triangular) of the objective f(x)."""
    n, w, S = inst.n, inst.w, inst.S
    lin = -inst.alpha * (S @ w) + inst.gamma * inst.stewardship
    Q = np.zeros((n, n))
    redundancy = (S * w) @ S.T        # R[d,e] = sum_j w_j S_dj S_ej
    for d, e in combinations(range(n), 2):
        Q[d, e] = inst.alpha * redundancy[d, e] + inst.beta * inst.T[d, e] + inst.delta * inst.D[d, e]
    return lin, Q


def build_program(inst: Instance) -> QuadraticProgram:
    """Constrained model as a Qiskit QuadraticProgram."""
    lin, Q = qubo_coefficients(inst)
    qp = QuadraticProgram("antibiotic_combination")
    for s in SHORT:
        qp.binary_var(s)
    quad = {(SHORT[d], SHORT[e]): Q[d, e] for d, e in combinations(range(inst.n), 2) if Q[d, e] != 0}
    qp.minimize(linear={SHORT[d]: lin[d] for d in range(inst.n)}, quadratic=quad)
    qp.linear_constraint({s: 1 for s in SHORT}, "==", inst.k, "regimen_size")
    qp.linear_constraint({SHORT[i]: 1 for i in inst.nephro_idx}, "<=", 1, "max_one_nephrotoxic")
    return qp


def default_penalty(inst: Instance) -> float:
    """Smallest 'safe' penalty: larger than any objective change caused by flipping one bit."""
    lin, Q = qubo_coefficients(inst)
    Qs = Q + Q.T
    max_flip = max(abs(lin[d]) + np.abs(Qs[d]).sum() for d in range(inst.n))
    return float(np.ceil(1.5 * max_flip * 10) / 10)


def to_qubo(inst: Instance, penalty: float | None = None):
    """Unconstrained QUBO (constraints moved into the objective as penalties)."""
    qp = build_program(inst)
    if penalty is None:
        penalty = default_penalty(inst)
    conv = QuadraticProgramToQubo(penalty=penalty)
    qubo = conv.convert(qp)
    return qp, qubo, conv, penalty


# ---------- evaluation helpers -------------------------------------------------

def surrogate_objective(inst: Instance, x) -> float:
    lin, Q = qubo_coefficients(inst)
    x = np.asarray(x, dtype=float)
    return float(lin @ x + x @ Q @ x)


def true_coverage(inst: Instance, x) -> float:
    """Exact prevalence-weighted coverage 1 - prod(1 - S) (not the QUBO approximation)."""
    x = np.asarray(x, dtype=bool)
    miss = np.prod(1 - inst.S[x], axis=0) if x.any() else np.ones(inst.S.shape[1])
    return float(inst.w @ (1 - miss))


def true_objective(inst: Instance, x) -> float:
    """Same objective as the QUBO but with exact (non-linear) coverage."""
    x = np.asarray(x, dtype=float)
    pen = 0.0
    for d, e in combinations(range(inst.n), 2):
        pen += x[d] * x[e] * (inst.beta * inst.T[d, e] + inst.delta * inst.D[d, e])
    return float(-inst.alpha * true_coverage(inst, x) + pen + inst.gamma * inst.stewardship @ x)


def is_feasible(inst: Instance, x) -> bool:
    x = np.asarray(x, dtype=int)
    return int(x.sum()) == inst.k and int(x[inst.nephro_idx].sum()) <= 1


def brute_force(inst: Instance):
    """Exact reference: enumerate all 2^n assignments (1024 for n = 10)."""
    rows = []
    for bits in product([0, 1], repeat=inst.n):
        x = np.array(bits)
        if is_feasible(inst, x):
            rows.append((surrogate_objective(inst, x), true_objective(inst, x), x))
    rows.sort(key=lambda r: r[0])
    return rows


def regimen(x) -> str:
    return " + ".join(SHORT[i] for i, v in enumerate(x) if round(v) == 1) or "(none)"
