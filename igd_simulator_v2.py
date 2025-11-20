#!/usr/bin/env python3
"""
IGD simulator v2: unified CLI for TFIM, 1+1D CFT, and SYK-like toy models.

Usage examples:
  python igd_simulator_v2.py --model tfim --mode both
  python igd_simulator_v2.py --model cft  --mode both --cft_c 0.5 --compare_tfim_critical
  python igd_simulator_v2.py --model syk  --mode dynamic
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from models import tfim, cft_1p1, syk_toy


def run_tfim(args):
    N = args.N
    J = args.J
    h = args.h
    fig_dir = Path(args.figdir)

    print(f"[IGD-TFIM] Running with N={N}, J={J}, h={h}")

    # Build Hamiltonian and compute spectrum
    H = tfim.tfim_hamiltonian(N, J, h)
    E0, psi_gs, eigvals, eigvecs = tfim.ground_state(H)
    eigvals_sorted = np.sort(eigvals.real)
    if len(eigvals_sorted) > 1:
        gap = float(eigvals_sorted[1] - eigvals_sorted[0])
    else:
        gap = 0.0

    # Theoretical IGD length/velocity for TFIM
    if J > 0 and h > 0:
        xi_theory = 1.0 / np.log(J / h)
        v_igd = 2.0 * min(J, h)
    else:
        xi_theory = float("nan")
        v_igd = float("nan")

    print(f"[IGD-TFIM] Ground-state energy E0 ≈ {E0:.6f}")
    print(f"[IGD-TFIM] Gap Δ ≈ {gap:.6f}")
    print(f"[IGD-TFIM] ξ_IGD (theory) ≈ {xi_theory:.4f}")
    print(f"[IGD-TFIM] v_IGD (theory) ≈ {v_igd:.4f}")

    # Static IGD observables: mutual information vs distance + block entropies
    if args.mode in ("static", "both"):
        d, I_d_bits, _E0 = tfim.run_static(N=N, J=J, h=h, fig_dir=str(fig_dir), prefix="TFIM")
        print("[IGD-TFIM] Static mutual information I(d) [bits]:")
        for dist, Ival in zip(d, I_d_bits):
            print(f"  d={int(dist)}: I ≈ {Ival:.4e}")

        ells, S_bits, _E0 = tfim.plot_block_entropies(N=N, J=J, h=h, fig_dir=str(fig_dir), prefix="TFIM")
        print("[IGD-TFIM] Ground-state block entropies S(ell) [bits]:")
        for ell, Sval in zip(ells, S_bits):
            print(f"  ell={int(ell)}: S ≈ {Sval:.4e}")
        print("[IGD-TFIM] Block entropy plot saved in figures/")

    # Dynamic IGD observables: entanglement and I(d,t)
    if args.mode in ("dynamic", "both"):
        times, S_half_bits, d_vals, I_dt_bits = tfim.run_dynamic(
            N=N, J=J, h=h, t_max=args.tmax, num_t=args.num_t,
            fig_dir=str(fig_dir), prefix="TFIM"
        )
        print("[IGD-TFIM] Dynamic half-chain entropy S_A(t) [bits]:")
        print(f"  S_A(0) ≈ {S_half_bits[0]:.4e}, S_A(t_end) ≈ {S_half_bits[-1]:.4e}")
        print("[IGD-TFIM] I(d,t) heatmap written to figures/")


def run_cft(args):
    fig_dir = Path(args.figdir)
    c = args.cft_c

    print(f"[IGD-CFT] Running analytic CFT demo with central charge c={c}")

    # Static CFT entanglement
    if args.mode in ("static", "both"):
        ell, S = cft_1p1.run_static(fig_dir=str(fig_dir), prefix="CFT", c=c)
        print("[IGD-CFT] Static entanglement S(ℓ) ~ (c/3) log(ℓ/a):")
        print(f"  S(ℓ_min) ≈ {S[0]:.4f}, S(ℓ_max) ≈ {S[-1]:.4f}")

    # Quench entanglement
    if args.mode in ("dynamic", "both"):
        t, S_t = cft_1p1.run_dynamic(fig_dir=str(fig_dir), prefix="CFT", c=c)
        print("[IGD-CFT] Quench entanglement S_A(t):")
        print(f"  S_A(t_min) ≈ {S_t[0]:.4f}, S_A(t_max) ≈ {S_t[-1]:.4f}")

    # Optional: compare with critical TFIM for block entropies
    if args.compare_tfim_critical:
        N = args.N
        print(f"[IGD-CFT] Comparing CFT (c={c}) with critical TFIM block entropies (N={N}, J=1, h=1)")
        Jc = 1.0
        hc = 1.0
        ells_tfim, S_tfim_bits, E0 = tfim.block_entropies_ground_state(N=N, J=Jc, h=hc)

        # CFT prediction for the same block lengths (convert to bits)
        a = 1.0
        ells_cft = ells_tfim
        S_cft_nats = (c / 3.0) * np.log(ells_cft / a)
        S_cft_bits = S_cft_nats / np.log(2.0)

        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(ells_tfim, S_tfim_bits, 'o-', label='TFIM (critical, ground state)')
        plt.plot(ells_cft, S_cft_bits, 's--', label=f'CFT prediction (c={c})')
        plt.xlabel('Block length ell')
        plt.ylabel('Entanglement entropy S(ell) [bits]')
        plt.title(f'Critical TFIM vs CFT (c={c})')
        plt.legend()
        plt.tight_layout()
        out = fig_dir / f"CFT_TFIM_Critical_BlockEntropy_Comparison_N{N}.png"
        plt.savefig(out)
        plt.close()
        print(f"[IGD-CFT] Comparison plot saved to {out}")


def run_syk(args):
    fig_dir = Path(args.figdir)
    Nq = args.Nq

    print(f"[IGD-SYKtoy] Running chaotic toy model with Nq={Nq} qubits")

    times, S_half_bits = syk_toy.run_demo(
        mode="dynamic", fig_dir=str(fig_dir), Nq=Nq, t_max=args.tmax, num_t=args.num_t
    )
    print("[IGD-SYKtoy] Half-system entropy S_A(t) [bits]:")
    print(f"  S_A(0) ≈ {S_half_bits[0]:.4e}, S_A(t_end) ≈ {S_half_bits[-1]:.4e}")


def main():
    parser = argparse.ArgumentParser(
        description="IGD simulator v2: TFIM, CFT, SYK-like toy."
    )
    parser.add_argument(
        "--model", choices=["tfim", "cft", "syk"], default="tfim",
        help="Which model to run."
    )
    parser.add_argument(
        "--mode", choices=["static", "dynamic", "both"], default="both",
        help="Which observables to compute."
    )
    parser.add_argument(
        "--figdir", default="figures",
        help="Directory for output figures."
    )

    # TFIM-specific
    parser.add_argument("--N", type=int, default=8, help="Number of TFIM spins (or matching size for CFT-TFIM comparison).")
    parser.add_argument("--J", type=float, default=1.0, help="TFIM coupling J.")
    parser.add_argument("--h", type=float, default=0.5, help="TFIM field h.")
    parser.add_argument("--tmax", type=float, default=6.0, help="Max time for dynamics.")
    parser.add_argument("--num_t", type=int, default=60, help="Number of time points.")

    # CFT-specific
    parser.add_argument("--cft_c", type=float, default=1.0, help="CFT central charge c.")
    parser.add_argument(
        "--compare_tfim_critical", action="store_true",
        help="If set (for --model cft), compare CFT entanglement with critical TFIM block entropies (J=1, h=1)."
    )

    # SYK toy-specific
    parser.add_argument("--Nq", type=int, default=6, help="Number of qubits in SYK toy.")

    args = parser.parse_args()

    if args.model == "tfim":
        run_tfim(args)
    elif args.model == "cft":
        run_cft(args)
    elif args.model == "syk":
        run_syk(args)
    else:
        raise ValueError(f"Unknown model: {args.model!r}")


if __name__ == "__main__":
    main()
