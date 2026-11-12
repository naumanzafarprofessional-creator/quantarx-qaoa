# Results summary (auto-generated)

Qubits: 10 | feasible regimens: 112 / 1024 | default penalty: 7.1

## Classical references (ICU sepsis)

| Rank | Regimen | QUBO objective | True objective | True coverage |
|---|---|---|---|---|
| 1 | VAN + CIP + MTZ | -0.9228 | -0.9228 | 0.795 |
| 2 | FEP + VAN + MTZ | -0.9047 | -0.9061 | 0.791 |
| 3 | PTZ + LZD + GEN | -0.8899 | -0.8931 | 0.895 |
| 4 | PTZ + LZD + AZM | -0.8725 | -0.9110 | 0.841 |
| 5 | CRO + VAN + CIP | -0.8683 | -0.9089 | 0.786 |
| 6 | PTZ + GEN + AZM | -0.8649 | -0.8670 | 0.778 |
| 7 | FEP + LZD + MTZ | -0.8647 | -0.8661 | 0.791 |
| 8 | PTZ + LZD + MTZ | -0.8618 | -0.8757 | 0.797 |
| 9 | MEM + VAN + AZM | -0.8597 | -0.9003 | 0.900 |
| 10 | PTZ + LZD + CIP | -0.8583 | -0.8968 | 0.925 |

- Exact eigensolver ground state: **VAN + CIP + MTZ**
- Greedy heuristic: **PTZ + LZD + GEN** (objective -0.8899)
- Optimum of the exact non-linear objective: **VAN + CIP + MTZ**

## Quantum benchmark (mean ± std over seeds, exact probabilities at optimised angles)

| Method | p | P(optimal) | ×random(weight-3) | P(top-5) | P(feasible) | approx. ratio | CX | depth |
|---|---|---|---|---|---|---|---|---|
| CVaR-QAOA (penalty, a=0.1) | 1 | 0.0043 ± 0.0001 | 0.5× | 0.021 | 0.471 | 0.750 | 90 | 59 |
| CVaR-QAOA (penalty, a=0.1) | 2 | 0.0030 ± 0.0005 | 0.4× | 0.015 | 0.342 | 0.748 | 180 | 94 |
| CVaR-QAOA (penalty, a=0.1) | 3 | 0.0048 ± 0.0005 | 0.6× | 0.024 | 0.545 | 0.749 | 270 | 129 |
| QAOA (penalty) | 1 | 0.0035 ± 0.0001 | 0.4× | 0.018 | 0.388 | 0.749 | 90 | 59 |
| QAOA (penalty) | 2 | 0.0046 ± 0.0003 | 0.6× | 0.023 | 0.502 | 0.751 | 180 | 94 |
| QAOA (penalty) | 3 | 0.0058 ± 0.0002 | 0.7× | 0.029 | 0.635 | 0.752 | 270 | 129 |
| XY-CVaR-QAOA (a=0.1) | 1 | 0.0160 ± 0.0001 | 1.9× | 0.068 | 0.994 | 0.819 | 290 | 572 |
| XY-CVaR-QAOA (a=0.1) | 2 | 0.0249 ± 0.0034 | 3.0× | 0.089 | 0.937 | 0.768 | 394 | 635 |
| XY-CVaR-QAOA (a=0.1) | 3 | 0.0346 ± 0.0034 | 4.2× | 0.102 | 0.947 | 0.769 | 498 | 698 |
| XY-QAOA | 1 | 0.0127 ± 0.0004 | 1.5× | 0.066 | 0.984 | 0.834 | 290 | 572 |
| XY-QAOA | 2 | 0.0134 ± 0.0018 | 1.6× | 0.072 | 0.995 | 0.853 | 394 | 635 |
| XY-QAOA | 3 | 0.0149 ± 0.0022 | 1.8× | 0.077 | 0.995 | 0.871 | 498 | 698 |

Random guessing: P(opt) = 0.0010 over all bitstrings, 0.0083 over 3-drug regimens; P(feasible) = 0.109 over all bitstrings.

## Penalty sensitivity (standard QAOA, p = 2)

| Penalty | P(optimal) | P(feasible) |
|---|---|---|
| 0.25 | 0.0072 | 0.438 |
| 0.5 | 0.0052 | 0.453 |
| 1.0 | 0.0045 | 0.452 |
| 2.0 | 0.0045 | 0.473 |
| 4.0 | 0.0044 | 0.478 |
| 7.1 | 0.0048 | 0.524 |

## Coverage-approximation fidelity (alpha sweep)

| alpha | QUBO optimum | true optimum | agree |
|---|---|---|---|
| 0.5 | CRO + VAN + MTZ | CRO + VAN + MTZ | yes |
| 0.75 | VAN + CIP + MTZ | VAN + CIP + MTZ | yes |
| 1.0 | VAN + CIP + MTZ | VAN + CIP + MTZ | yes |
| 1.25 | VAN + CIP + MTZ | VAN + CIP + MTZ | yes |
| 1.5 | VAN + CIP + MTZ | VAN + CIP + MTZ | yes |
| 1.75 | VAN + CIP + MTZ | PTZ + LZD + CIP | no |
| 2.0 | PTZ + LZD + GEN | PTZ + LZD + CIP | no |
| 2.25 | PTZ + LZD + GEN | PTZ + LZD + CIP | no |
| 2.5 | PTZ + LZD + GEN | PTZ + LZD + CIP | no |
| 2.75 | PTZ + LZD + GEN | MEM + VAN + CIP | no |
| 3.0 | PTZ + LZD + GEN | MEM + VAN + CIP | no |

## Second scenario: community-acquired pneumonia

- Exact optimum: **CRO + AZM + MTZ**
- XY-CVaR-QAOA (a=0.1) p=3: P(optimal) = 0.0848, P(top-5) = 0.145, P(feasible) = 0.982
