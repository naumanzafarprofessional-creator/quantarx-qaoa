# QuantaRx – 3-minute demo video script

About 430 spoken words, which takes roughly 2:50 at a normal speaking pace.
**[Screen]** lines say what to show. **Say** lines are the narration.

Before you record:

- Run `python run_experiments.py --quick` once, so the figures in `results/` are fresh.
- Open the README on GitHub in one tab and a terminal in the project folder in another.
- Have the six PNGs from `results/` ready to show full screen.

---

## 0:00 – 0:20 · The problem

**[Screen]** Title card: *QuantaRx: choosing antibiotic combinations with constraint-preserving QAOA*. Then show `results/fig_problem.png`.

**Say:**
> A patient arrives in the ICU with sepsis, and the pathogen isn't known yet. The doctor has to start a three-drug regimen right away. It has to cover the likely bugs, avoid toxic pairs such as vancomycin with gentamicin, and not waste last-line antibiotics. With ten drugs, there are 120 possible regimens, and they interact in pairs. That makes it a combinatorial optimization problem, and we solve it on a quantum computer.

## 0:20 – 0:50 · The formulation

**[Screen]** README section on the formulation (the LaTeX objective), then the mermaid architecture diagram.

**Say:**
> Each drug is one qubit, so ten qubits in total. The objective rewards how much of the pathogen mix the regimen covers. Coverage is really a product, so we expand it by inclusion–exclusion. The pairwise term turns out to be the redundancy between drugs, which gives us an exact QUBO. On top of that we add pairwise toxicity, a penalty for duplicate drug classes, and a stewardship cost. There are two constraints: exactly three drugs, and at most one kidney-toxic drug. We build it as a Qiskit `QuadraticProgram`. Our tests check that the QUBO's ground state is the true best regimen.

## 0:50 – 1:30 · The quantum method

**[Screen]** `drugcombo/solvers.py`: scroll to `xy_qaoa_ansatz`, then `drugcombo/dicke.py`.

**Say:**
> The textbook approach is QAOA with the regimen-size constraint written as a penalty. It works badly, because most of its samples are not valid three-drug regimens at all. So we enforce the constraint in the circuit itself. First, a Dicke-state circuit puts the qubits in an equal superposition of every three-drug regimen. Then an XY mixer swaps drugs in and out without ever changing how many are chosen. That way, the circuit only ever explores valid regimens. On top of this we use CVaR to focus training on the best samples, INTERP warm starts from layer to layer, and several random starting points. Everything runs on Qiskit 2 primitives.

## 1:30 – 2:05 · Live run and results

**[Screen]** Terminal: run `python run_experiments.py --quick`. Speed up the recording while it runs. Then show `results/fig_benchmark.png` and `results/fig_distribution.png`.

**Say:**
> Here's the pipeline running end to end. The classical exact solver finds the optimum: vancomycin, ciprofloxacin and metronidazole. A greedy heuristic picks a different regimen, which only ranks third. Here are the results at three QAOA layers. Penalty QAOA produces a valid regimen only 64 percent of the time. Our XY-QAOA does so over 99 percent of the time. XY with CVaR samples the optimal regimen 3.5 percent of the time. That's six times penalty QAOA, four times random guessing among valid regimens, and 35 times random bitstrings.

## 2:05 – 2:35 · Noise study

**[Screen]** `results/fig_noise.png`.

**Say:**
> Real hardware is noisy, so we simulated it in Qiskit Aer with a realistic noise model scaled from zero up to today's error rates. We also routed the circuits onto IBM's heavy-hex layout. Since every valid answer has exactly three drugs, any shot with a different count must be an error, and we throw it away. With that filter, our advantage survives today's noise on all-to-all hardware. On heavy-hex, it survives up to about half of today's error rate. Routing doubles the two-qubit gate count, and that is what our method is currently up against.

## 2:35 – 3:00 · Limitations and close

**[Screen]** README "Limitations" section, then the GitHub repo URL.

**Say:**
> To be honest about the limits: our data is synthetic, the instance is small enough to brute-force, and we make no claim of quantum advantage. What we show is a working pattern. Hard clinical constraints are built into the quantum circuit itself, and the approach is benchmarked against classical baselines and tested under noise. The code, the tests and the full report are on GitHub. Thanks for watching.

---

### Tips

- If `--quick` takes too long to show live, record it beforehand and play it at 4–8× speed.
- Keep each figure on screen for at least 5 seconds.
- If you need to cut time, shorten the method section first. The results and noise sections score the most points under *Quantum utilization*.
