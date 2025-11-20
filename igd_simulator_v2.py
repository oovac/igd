#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

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
    gap = float(eigvals_sorted[1] - eigvals_sorted[0]) if len(eigvals_sorted) > 1 else 0.0

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

    # Static IGD observables: mutual information vs distance
    if args.mode in ("static", "both"):
        d, I_d_bits, _E0 = tfim.run_static(N=N, J=J, h=h, fig_dir=str(fig_dir), prefix="TFIM")
        print("[IGD-TFIM] Static mutual information I(d) [bits]:")
        for dist, Ival in zip(d, I_d_bits):
            print(f"  d={int(dist)}: I ≈ {Ival:.4e}")

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

    if args.mode in ("static", "both"):
        ell, S = cft_1p1.run_static(fig_dir=str(fig_dir), prefix="CFT", c=c)
        print("[IGD-CFT] Static entanglement S(ℓ) ~ (c/3) log(ℓ/a):")
        print(f"  S(ℓ_min) ≈ {S[0]:.4f}, S(ℓ_max) ≈ {S[-1]:.4f}")

    if args.mode in ("dynamic", "both"):
        t, S_t = cft_1p1.run_dynamic(fig_dir=str(fig_dir), prefix="CFT", c=c)
        print("[IGD-CFT] Quench entanglement S_A(t):")
        print(f"  S_A(t_min) ≈ {S_t[0]:.4f}, S_A(t_max) ≈ {S_t[-1]:.4f}")


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
    parser.add_argument("--N", type=int, default=8, help="Number of TFIM spins.")
    parser.add_argument("--J", type=float, default=1.0, help="TFIM coupling J.")
    parser.add_argument("--h", type=float, default=0.5, help="TFIM field h.")
    parser.add_argument("--tmax", type=float, default=6.0, help="Max time for dynamics.")
    parser.add_argument("--num_t", type=int, default=60, help="Number of time points.")

    # CFT-specific
    parser.add_argument("--cft_c", type=float, default=1.0, help="CFT central charge c.")

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
