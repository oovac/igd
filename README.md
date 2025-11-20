# IGD Multimodel Demo: TFIM, CFT, SYK

This repository is a unified demonstration of Information–Geometric Dynamics (IGD)
across several models:

- **TFIM** (1D transverse-field Ising model, gapped phase, exact diagonalization)
- **CFT** (1+1D conformal field theory, analytic entanglement formulas)
- **SYK toy** (chaotic all-to-all random Hamiltonian as a SYK-like toy model)

All three modes share the same base philosophy:

- Start from a simple initial state (ground state or product state).
- Evolve or evaluate information-theoretic quantities (entropy, mutual information).
- Visualize locality / light-cones / chaos in an IGD-friendly language.

## Structure

- `run_igd_demo.py` — main entry point, with CLI switch `--model {tfim,cft,syk}`
  and `--mode {static,dynamic,both}`.
- `models/` — model-specific implementations:
  - `tfim.py` — static and dynamic ED demo in 1D TFIM (N=8).
  - `cft_1p1.py` — analytic entropy curves for 1+1D CFT (Calabrese–Cardy style).
  - `syk_toy.py` — small random all-to-all Hamiltonian as a SYK-like chaotic toy.
- `paper/` — mini-preprints / write-ups (TFIM static & dynamic included).
- `figures/` — output plots from the demos.
- `theory/` — IGD axiomatic and spin-chain program LaTeX files (if present).

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`

Install via:

```bash
pip install numpy matplotlib
```

## Quick start

From the repository root:

```bash
python run_igd_demo.py --model tfim --mode both
python run_igd_demo.py --model cft --mode both
python run_igd_demo.py --model syk --mode dynamic
```

The scripts will produce PNG figures in the `figures/` directory.

TFIM uses small exact diagonalization (N=8) and is meant as a clear, fully explicit
IGD example. CFT and SYK toy modes are lightweight analytic / random-matrix demos
that fit into the same conceptual pipeline.
