"""
Illustrative problem instance: empiric antibiotic combination selection.

IMPORTANT: every number below is a *synthetic, literature-inspired* value made up
for this hackathon demo (a fictional hospital "antibiogram"). It is NOT clinical
guidance and must not be used for treatment decisions.

The instance has 10 candidate drugs -> 10 binary decision variables -> 10 qubits.
"""
from dataclasses import dataclass, field
import numpy as np

DRUGS = [
    "Piperacillin-tazobactam",  # 0  beta-lactam
    "Meropenem",                # 1  carbapenem (reserve agent)
    "Ceftriaxone",              # 2  beta-lactam
    "Cefepime",                 # 3  beta-lactam
    "Vancomycin",               # 4  glycopeptide, nephrotoxic
    "Linezolid",                # 5  oxazolidinone
    "Gentamicin",               # 6  aminoglycoside, nephrotoxic
    "Ciprofloxacin",            # 7  fluoroquinolone, QT-prolonging
    "Azithromycin",             # 8  macrolide, QT-prolonging
    "Metronidazole",            # 9  nitroimidazole
]
SHORT = ["PTZ", "MEM", "CRO", "FEP", "VAN", "LZD", "GEN", "CIP", "AZM", "MTZ"]

PATHOGENS = [
    "MRSA", "S. pneumoniae", "E. coli", "Klebsiella",
    "P. aeruginosa", "Acinetobacter", "Anaerobes", "Atypicals",
]

# Susceptibility matrix S[d, j]: fraction of isolates of pathogen j that drug d is
# active against (synthetic local antibiogram).
S = np.array([
    #  MRSA  Spn   Eco   Kpn   Psa   Aba   Ana   Aty
    [0.00, 0.90, 0.85, 0.70, 0.80, 0.20, 0.95, 0.00],  # PTZ
    [0.00, 0.95, 0.98, 0.90, 0.80, 0.40, 0.95, 0.00],  # MEM
    [0.00, 0.95, 0.80, 0.60, 0.00, 0.00, 0.10, 0.00],  # CRO
    [0.00, 0.95, 0.85, 0.70, 0.80, 0.10, 0.10, 0.00],  # FEP
    [0.98, 0.95, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00],  # VAN
    [0.98, 0.95, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00],  # LZD
    [0.30, 0.05, 0.85, 0.80, 0.75, 0.40, 0.00, 0.00],  # GEN
    [0.10, 0.60, 0.65, 0.70, 0.70, 0.30, 0.00, 0.90],  # CIP
    [0.05, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.95],  # AZM
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.98, 0.00],  # MTZ
])

# Pathogen prevalence weights for each clinical setting (each row sums to 1).
SCENARIOS = {
    # Hospital-acquired / ICU sepsis: resistant Gram-negatives and MRSA dominate
    "icu_sepsis":   np.array([0.25, 0.05, 0.15, 0.15, 0.20, 0.05, 0.10, 0.05]),
    # Community-acquired pneumonia: pneumococcus and atypicals dominate
    "community_pneumonia": np.array([0.05, 0.40, 0.05, 0.05, 0.03, 0.00, 0.02, 0.40]),
}

# Pairwise adverse-interaction severity T[d, e] in [0, 1] (symmetric).
_T_PAIRS = {
    ("VAN", "GEN"): 1.0,   # additive nephrotoxicity
    ("VAN", "PTZ"): 0.6,   # reported higher acute-kidney-injury risk
    ("CIP", "AZM"): 0.8,   # additive QT prolongation
    ("GEN", "CIP"): 0.2,
    ("VAN", "FEP"): 0.15,
    ("LZD", "CIP"): 0.1,
}

# Same-class duplication (adds little benefit, more selection pressure toward resistance).
_CLASSES = {
    "beta_lactam": ["PTZ", "MEM", "CRO", "FEP"],
    "anti_mrsa":   ["VAN", "LZD"],
}
CLASS_DUP_PENALTY = 1.0

# Antimicrobial-stewardship cost per drug (reserve / broad-spectrum agents cost more).
STEWARDSHIP = np.array([0.10, 0.30, 0.02, 0.10, 0.10, 0.20, 0.15, 0.15, 0.05, 0.02])

# Drugs that may not be combined with each other (at most one allowed).
NEPHROTOXIC = ["VAN", "GEN"]


def _sym_from_pairs(pairs: dict) -> np.ndarray:
    n = len(SHORT)
    M = np.zeros((n, n))
    for (a, b), v in pairs.items():
        i, j = SHORT.index(a), SHORT.index(b)
        M[i, j] = M[j, i] = v
    return M


@dataclass
class Instance:
    scenario: str = "icu_sepsis"
    k: int = 3                     # number of drugs in the regimen
    alpha: float = 1.5             # weight on pathogen coverage
    beta: float = 0.4              # weight on adverse interactions
    delta: float = 0.3             # weight on same-class duplication
    gamma: float = 1.0             # weight on stewardship cost
    S: np.ndarray = field(default_factory=lambda: S.copy())
    stewardship: np.ndarray = field(default_factory=lambda: STEWARDSHIP.copy())

    @property
    def n(self) -> int:
        return len(DRUGS)

    @property
    def w(self) -> np.ndarray:
        return SCENARIOS[self.scenario]

    @property
    def T(self) -> np.ndarray:
        return _sym_from_pairs(_T_PAIRS)

    @property
    def D(self) -> np.ndarray:
        n = self.n
        M = np.zeros((n, n))
        for members in _CLASSES.values():
            idx = [SHORT.index(m) for m in members]
            for a in idx:
                for b in idx:
                    if a != b:
                        M[a, b] = CLASS_DUP_PENALTY
        return M

    @property
    def nephro_idx(self) -> list[int]:
        return [SHORT.index(m) for m in NEPHROTOXIC]
