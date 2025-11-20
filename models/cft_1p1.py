import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def static_entanglement(c=1.0, L_max=50, a=1.0):
    """S(ℓ) ~ (c/3) log(ℓ/a) up to an additive constant (set to zero)."""
    ell = np.arange(1, L_max+1)
    S = (c/3.0) * np.log(ell/a)
    return ell, S

def quench_entanglement(c=1.0, ell=40.0, v=1.0, a=1.0, t_min=0.1, t_max=50.0, num_t=200):
    """Very simple Calabrese–Cardy-style quench entropy for an interval of length ℓ.

    For t < ℓ/(2v): S(t) ~ (c/3) log(2vt/a)
    For t > ℓ/(2v): S(t) saturates to (c/3) log(ℓ/a)

    This is meant as an analytic CFT demo, not a full calculation.
    """
    t = np.linspace(t_min, t_max, num_t)
    S = np.zeros_like(t)
    t_star = ell / (2.0 * v)
    S_sat = (c/3.0) * np.log(ell/a)
    for i, ti in enumerate(t):
        if ti < t_star:
            S[i] = (c/3.0) * np.log(2.0 * v * ti / a)
        else:
            S[i] = S_sat
    return t, S

def run_static(fig_dir="figures", prefix="CFT", c=1.0):
    ell, S = static_entanglement(c=c)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(ell, S)
    plt.xlabel('Interval length ℓ (arbitrary units)')
    plt.ylabel('Entanglement entropy S(ℓ)')
    plt.title(f'1+1D CFT static entanglement (c={{c}})')
    plt.tight_layout()
    out = fig_dir / f"{prefix}_Static_Entanglement.png"
    plt.savefig(out)
    plt.close()
    return ell, S

def run_dynamic(fig_dir="figures", prefix="CFT", c=1.0):
    ell = 40.0
    t, S = quench_entanglement(c=c, ell=ell)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(t, S)
    plt.xlabel('Time t (arbitrary units)')
    plt.ylabel('Entanglement entropy S_A(t)')
    plt.title(f'1+1D CFT global quench (interval ℓ={{ell}}, c={{c}})')
    plt.tight_layout()
    out = fig_dir / f"{prefix}_Quench_Entanglement.png"
    plt.savefig(out)
    plt.close()
    return t, S

def run_demo(mode="both", **kwargs):
    if mode in ("static", "both"):
        run_static(**kwargs)
    if mode in ("dynamic", "both"):
        run_dynamic(**kwargs)
