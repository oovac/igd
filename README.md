# IGD Multimodel Demo: TFIM, CFT, SYK (random + SYK4)

This repository is a unified demonstration of Information–Geometric Dynamics (IGD)
across three complementary model sectors:

- **TFIM** – 1D transverse-field Ising model (gapped phase, exact diagonalization),
  used as a clean T1 (area-law) testbed.
- **CFT** – 1+1D conformal field theory entanglement formulas (and comparison
  to critical TFIM), providing a bridge towards T2 (entanglement thermodynamics).
- **SYK** – chaotic models:
  - a random-Hermitian toy on \(N_q\) qubits;
  - a structured Majorana SYK\(_4\)-like model on \(N_f\) fermionic modes,
    used as a T3 (chaos / complexity) playground.

All three sectors share the same IGD philosophy:

- start from a simple initial state (ground state or product state);
- evolve or evaluate information-theoretic quantities (entropy, mutual information);
- visualise locality / light-cones / scrambling in an information–geometric language.

## 1. Quick start: run everything

From the repository root, simply run:

```bash
python run_igd_demo.py
```

With no arguments, the script will:

1. Run the TFIM demo (static + dynamic):
   - ground-state gap and IGD correlation length \(\xi_{\text{IGD}}\);
   - static mutual information \(I(d)\);
   - block entropies \(S_A(\ell)\) (T1 area-law diagnostic);
   - dynamical half-chain entropy \(S_A(t)\) and mutual-information light-cones \(I(d,t)\).
2. Run the CFT demo (with \(c = 1/2\)) and compare to a critical TFIM chain.
3. Run the SYK\(_4\) toy demo (Majorana SYK\(_4\)-like model) and plot
   half-system entanglement growth \(S_A(t)\) as a simple chaos / complexity proxy.

All figures are written to the `figures/` directory.

## 2. Advanced usage

The unified entry point is **only one file**:

```bash
run_igd_demo.py
```

You can still control what is run via command-line arguments.

### 2.1 TFIM mode (gapped / T1)

```bash
python run_igd_demo.py --model tfim --mode both
```

Options:

- `--N` – number of spins (default: 8).
- `--J` – Ising coupling (default: 1.0).
- `--h` – transverse field (default: 0.5, gapped ferromagnetic regime).
- `--tmax`, `--num_t` – time range and resolution for dynamics.
- `--figdir` – output directory for PNG figures (default: `figures`).

### 2.2 CFT mode (critical / T1–T2 bridge)

```bash
python run_igd_demo.py --model cft --mode both --cft_c 0.5 --compare_tfim_critical
```

Options:

- `--cft_c` – central charge \(c\) of the 1+1D CFT (default: 1.0).
- `--compare_tfim_critical` – if set, computes critical TFIM block entropies
  (with \(J = h = 1\)) and compares them to the CFT prediction on the same
  interval lengths.
- `--N` – TFIM size used for the comparison plot.

### 2.3 SYK mode (chaotic / T3 sector)

```bash
# Random-Hermitian toy (simple chaotic benchmark)
python run_igd_demo.py --model syk --mode dynamic --syk_mode random --Nq 6

# Structured Majorana SYK4-like model (T3 playground)
python run_igd_demo.py --model syk --mode dynamic --syk_mode syk4 --Nf 4
```

Options:

- `--syk_mode` – `random` or `syk4`.
- `--Nq` – number of qubits for the random-Hermitian toy (Hilbert dimension \(2^{N_q}\)).
- `--Nf` – number of fermionic modes for the SYK\(_4\) toy (Hilbert dimension \(2^{N_f}\)).
- `--tmax`, `--num_t` – time range and resolution for dynamics.
- `--seed` – random seed for the SYK sector.

In all SYK modes the code computes half-system entanglement entropy \(S_A(t)\)
as a function of time; in the SYK\(_4\) mode this is the main T3 diagnostic.

## 3. Theory files and structure

The `theory/` directory contains the LaTeX theory notes:

- `IGD_Axiomatic_Foundation_v3.0.tex` – current axiomatic foundation (A0–A10).
- `IGD_Core_Dynamics_and_Theorems_v3.1.tex` – core IGD dynamics + overview of T1–T3.
- `IGD_T1_AreaLaw_SpinChains_v1.0.tex` and `IGD_Lattice_SpinChains_v1.0.tex` – T1
  and its realisation in gapped spin chains (TFIM).
- `IGD_CFT_1p1_v1.0.tex` – CFT (1+1D) entanglement + mapping to lattice models.
- `IGD_Chaos_Complexity_T3_v1.0.tex` and `IGD_SYK_Chaos_v1.0.tex` – T3 (chaos / complexity)
  and the SYK\(_4\) playground.
- `IGD_Multimodel_Demo_Documentation_v1.0.tex` – more detailed mapping between
  simulator modes and spacetime theorems.

Legacy / historical files that can be kept for reference (but are not needed
for current work):

- `axioms_formulas_v1.tex`, `igd_base_v1.tex` – early versions of the axioms.
- `IGD_Core_Dynamics_and_Theorems_v3.0.tex` – older core-note version superseded
  by v3.1.

The `paper/` directory contains mini–preprint drafts for TFIM static and
dynamical IGD analyses.

## 4. Requirements

See `requirements.txt` for Python dependencies (NumPy, Matplotlib and standard
scientific stack). For small system sizes (N up to 8–10) everything runs
comfortably on a laptop.

## 5. Repository cleanliness

The canonical way to run the demo is now:

```bash
python run_igd_demo.py
```

The older helper script `igd_simulator_v2.py` has been removed to avoid
duplication and confusion: all of its functionality has been merged into
`run_igd_demo.py`.
