"""Correctness checks: python -m pytest -q   (or: python tests/test_model.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from qiskit.quantum_info import Statevector

from drugcombo.data import Instance
from drugcombo.dicke import dicke_state
from drugcombo.formulation import (brute_force, to_qubo, surrogate_objective, true_coverage,
                                   is_feasible)
from drugcombo.solvers import exact_eigensolver, xy_qaoa_ansatz, xy_hamiltonian, normalise, decode, interp


def test_qubo_has_no_slack_qubits():
    _, qubo, _, _ = to_qubo(Instance())
    assert qubo.get_num_vars() == 10


def test_qubo_ground_state_is_constrained_optimum():
    """Penalties are large enough: ground state of the QUBO == brute-force feasible optimum."""
    for sc in ["icu_sepsis", "community_pneumonia"]:
        inst = Instance(scenario=sc)
        x_eig, _ = exact_eigensolver(inst)
        assert np.array_equal(x_eig, brute_force(inst)[0][2])


def test_qubo_objective_matches_qiskit_program():
    inst = Instance()
    qp, qubo, _, _ = to_qubo(inst)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, inst.n)
        if is_feasible(inst, x):   # on feasible points the penalties vanish
            assert abs(qubo.objective.evaluate(x) - surrogate_objective(inst, x)) < 1e-9


def test_second_order_coverage_is_lower_bound():
    """Inclusion-exclusion truncated at order 2 is a lower bound on true coverage (Bonferroni)."""
    inst = Instance()
    for _, _, x in brute_force(inst):
        S = inst.S[x.astype(bool)]
        approx = inst.w @ (S.sum(0) - sum(S[i] * S[j] for i in range(len(S)) for j in range(i + 1, len(S))))
        assert approx <= true_coverage(inst, x) + 1e-12


def test_dicke_state_exact():
    for n, k in [(4, 2), (6, 3), (10, 3)]:
        sv = Statevector(dicke_state(n, k)).data
        tgt = np.array([1.0 if bin(i).count("1") == k else 0.0 for i in range(2 ** n)])
        tgt /= np.linalg.norm(tgt)
        assert abs(abs(np.vdot(tgt, sv)) ** 2 - 1) < 1e-9


def test_xy_ansatz_preserves_regimen_size():
    """Whatever the angles, XY-QAOA never leaves the 'exactly k drugs' subspace."""
    inst = Instance()
    op, _ = xy_hamiltonian(inst)
    qc = xy_qaoa_ansatz(normalise(op), inst.n, inst.k, reps=2)
    rng = np.random.default_rng(1)
    pr = np.abs(Statevector(qc.assign_parameters(rng.uniform(-3, 3, qc.num_parameters))).data) ** 2
    leak = sum(p for i, p in enumerate(pr) if decode(i, inst.n).sum() != inst.k)
    assert leak < 1e-9


def test_xy_hamiltonian_ground_state_in_subspace_is_optimum():
    inst = Instance()
    op, _ = xy_hamiltonian(inst)
    diag = np.real(op.to_matrix(sparse=True).diagonal())
    idx = [i for i in range(2 ** inst.n) if bin(i).count("1") == inst.k]
    best = min(idx, key=lambda i: diag[i])
    assert np.array_equal(decode(best, inst.n), brute_force(inst)[0][2])


def test_interp_shapes():
    p = np.array([0.5, 0.2, 0.1, 0.4])       # p = 2: betas [0.5, 0.2], gammas [0.1, 0.4]
    q = interp(p)
    assert q.shape == (6,)
    assert np.isclose(q[0], 0.5) and np.isclose(q[3], 0.1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
