"""
Deterministic Dicke-state preparation |D(n,k)> = uniform superposition of all n-bit
strings with exactly k ones, using O(n*k) gates
(Bartschi & Eidenbenz, "Deterministic Preparation of Dicke States", FCT 2019).

Used as the QAOA initial state so the search starts inside the feasible
"exactly k drugs" subspace instead of over all 2^n bitstrings.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate


def _scs(l: int, k: int) -> QuantumCircuit:
    """Split-and-cyclic-shift unitary SCS_{l,k} on k+1 qubits.
    Local qubit m corresponds to paper qubit (l - k + m); local k is paper qubit l."""
    qc = QuantumCircuit(k + 1, name=f"SCS({l},{k})")
    top = k
    # two-qubit block on paper qubits (l-1, l)
    qc.cx(top - 1, top)
    qc.cry(2 * np.arccos(np.sqrt(1 / l)), top, top - 1)
    qc.cx(top - 1, top)
    # three-qubit blocks on paper qubits (l-i, l-i+1, l), i = 2..k
    for i in range(2, k + 1):
        a, b = top - i, top - i + 1
        qc.cx(a, top)
        qc.append(RYGate(2 * np.arccos(np.sqrt(i / l))).control(2), [top, b, a])
        qc.cx(a, top)
    return qc


def dicke_state(n: int, k: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    if k == 0:
        return qc
    qc.x(range(n - k, n))
    for l in range(n, k, -1):                   # SCS_{l,k} on paper qubits l-k..l
        qc.compose(_scs(l, k), qubits=range(l - k - 1, l), inplace=True)
    for l in range(k, 1, -1):                   # SCS_{l,l-1} on paper qubits 1..l
        qc.compose(_scs(l, l - 1), qubits=range(0, l), inplace=True)
    return qc
