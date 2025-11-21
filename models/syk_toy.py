import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def random_hermitian(D, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D,D)) + 1j*rng.normal(size=(D,D))
    H = (A + A.conj().T) / 2.0
    return H

def reduced_density_matrix(psi, Nq, subsystem):
    """Partial trace for Nq qubits, subsystem is list of qubit indices."""
    subsystem = list(subsystem)
    k = len(subsystem)
    all_sites = list(range(Nq))
    rest = [s for s in all_sites if s not in subsystem]
    psi_tensor = psi.reshape([2]*Nq)
    perm = subsystem + rest
    psi_perm = np.transpose(psi_tensor, axes=perm)
    dim_sub = 2**k
    dim_rest = 2**(Nq-k)
    psi_perm = psi_perm.reshape(dim_sub, dim_rest)
    rho = psi_perm @ psi_perm.conj().T
    return rho

def von_neumann_entropy(rho, tol=1e-12):
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(vals.real, 0, 1)
    vals = vals[vals > tol]
    if len(vals) == 0:
        return 0.0
    return float(-np.sum(vals * np.log(vals)))

def basis_state_z(Nq, up=True):
    dim = 2**Nq
    psi = np.zeros(dim, dtype=np.complex128)
    idx = 0 if up else (1 << Nq) - 1
    psi[idx] = 1.0
    return psi

def time_evolution_eig(eigvals, eigvecs, psi0, times):
    c0 = eigvecs.conj().T @ psi0
    states = []
    for t in times:
        phase = np.exp(-1j * eigvals * t)
        psi_t = eigvecs @ (phase * c0)
        states.append(psi_t)
    return states

def half_chain_entropy_vs_time(states, Nq):
    S_list = []
    half_sites = list(range(Nq//2))
    for psi in states:
        rho_half = reduced_density_matrix(psi, Nq, half_sites)
        S_half = von_neumann_entropy(rho_half)
        S_list.append(S_half)
    return np.array(S_list)

def run_dynamic(Nq=6, t_max=10.0, num_t=80, fig_dir="figures", prefix="SYK_Toy", seed=0):
    """SYK-like toy: random all-to-all Hermitian Hamiltonian on 2^Nq levels.

    This is *not* a faithful SYK implementation, but a small chaotic toy model
    that exhibits fast entanglement growth, serving as an IGD-style chaotic
    benchmark.
    """
    D = 2**Nq
    H = random_hermitian(D, seed=seed)
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = basis_state_z(Nq, up=True)
    times = np.linspace(0.0, t_max, num_t)
    states_t = time_evolution_eig(eigvals, eigvecs, psi0, times)
    S_half_t_nats = half_chain_entropy_vs_time(states_t, Nq)
    S_half_t_bits = S_half_t_nats / np.log(2.0)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(times, S_half_t_bits)
    plt.xlabel('Time t')
    plt.ylabel('Half-system entropy S_A(t) [bits]')
    plt.title(f'SYK-like toy (Nq={{Nq}} qubits): entanglement growth')
    plt.tight_layout()
    out = fig_dir / f"{prefix}_HalfEntropy_Nq{Nq}.png"
    plt.savefig(out)
    plt.close()
    return times, S_half_t_bits



def gamma_majorana_operators(Nf):
    """
    Construct 2*Nf Majorana operators gamma_i acting on Nf qubits via a
    Jordan--Wigner-like mapping. Returns a list of 2**Nf x 2**Nf matrices.
    """
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)

    gammas = []
    # gamma_{2j-1} = (prod_{k<j} Z_k) X_j
    # gamma_{2j}   = (prod_{k<j} Z_k) Y_j
    for j in range(Nf):
        prefix = [Z] * j
        suffix = [I] * (Nf - j - 1)

        # gamma_{2j-1}
        ops1 = prefix + [X] + suffix
        G1 = ops1[0]
        for op in ops1[1:]:
            G1 = np.kron(G1, op)
        gammas.append(G1)

        # gamma_{2j}
        ops2 = prefix + [Y] + suffix
        G2 = ops2[0]
        for op in ops2[1:]:
            G2 = np.kron(G2, op)
        gammas.append(G2)

    return gammas


def syk4_hamiltonian_majorana(Nf=4, J_scale=1.0, seed=None):
    """
    Build a small SYK4-like Hamiltonian using 2*Nf Majorana operators on Nf qubits.

    H = sum_{i<j<k<l} J_{ijkl} gamma_i gamma_j gamma_k gamma_l
    """
    rng = np.random.default_rng(seed)
    gammas = gamma_majorana_operators(Nf)
    M = len(gammas)  # 2*Nf
    dim = 2**Nf
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(M):
        for j in range(i+1, M):
            for k in range(j+1, M):
                for l in range(k+1, M):
                    J_ijkl = rng.normal(scale=J_scale / (M**1.5))
                    term = gammas[i] @ gammas[j] @ gammas[k] @ gammas[l]
                    H = H + J_ijkl * term
    H = (H + H.conj().T) / 2.0
    return H


def run_dynamic_syk4(Nf=4, t_max=10.0, num_t=80, fig_dir="figures",
                     prefix="SYK4", seed=0):
    """
    Time evolution for a small Majorana SYK4-like model built from Nf fermionic
    modes (2*Nf Majoranas). We encode the system on Nf qubits.
    """
    Nq = Nf
    H = syk4_hamiltonian_majorana(Nf=Nf, seed=seed)
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = basis_state_z(Nq, up=True)
    times = np.linspace(0.0, t_max, num_t)
    states_t = time_evolution_eig(eigvals, eigvecs, psi0, times)
    S_half_t_nats = half_chain_entropy_vs_time(states_t, Nq)
    S_half_t_bits = S_half_t_nats / np.log(2.0)

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(times, S_half_t_bits)
    plt.xlabel('Time t')
    plt.ylabel('Half-system entropy S_A(t) [bits]')
    plt.title(f'Majorana SYK4 toy (Nf={Nf}, Nq={Nq}): entanglement growth')
    plt.tight_layout()
    out = fig_dir / f"{prefix}_SYK4_HalfEntropy_Nf{Nf}.png"
    plt.savefig(out)
    plt.close()
    return times, S_half_t_bits


def run_demo(mode="dynamic", **kwargs):
    # For this toy we only implement dynamic mode.
    return run_dynamic(**kwargs)
