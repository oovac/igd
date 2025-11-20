import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def paulis():
    sx = np.array([[0., 1.], [1., 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    id2 = np.eye(2)
    return sx, sz, id2

def kron_n(ops):
    out = np.array([[1.]])
    for op in ops:
        out = np.kron(out, op)
    return out

def tfim_hamiltonian(N, J=1.0, h=0.5):
    sx, sz, id2 = paulis()
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.float64)
    # -J σ^z_i σ^z_{i+1}
    for i in range(N-1):
        ops = [id2]*N
        ops[i] = sz
        ops[i+1] = sz
        H -= J * kron_n(ops)
    # -h σ^x_i
    for i in range(N):
        ops = [id2]*N
        ops[i] = sx
        H -= h * kron_n(ops)
    return H

def ground_state(H):
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    return eigvals[idx], eigvecs[:, idx], eigvals, eigvecs

def reduced_density_matrix(psi, N, subsystem):
    """Vectorized partial trace: return ρ_sub for given subsystem list of sites."""
    subsystem = list(subsystem)
    k = len(subsystem)
    all_sites = list(range(N))
    rest = [s for s in all_sites if s not in subsystem]
    psi_tensor = psi.reshape([2]*N)
    perm = subsystem + rest
    psi_perm = np.transpose(psi_tensor, axes=perm)
    dim_sub = 2**k
    dim_rest = 2**(N-k)
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

def basis_state_z(N, up=True):
    dim = 2**N
    psi = np.zeros(dim, dtype=np.complex128)
    idx = 0 if up else (1 << N) - 1
    psi[idx] = 1.0
    return psi

def mutual_information(psi, N, i, j):
    rho_i = reduced_density_matrix(psi, N, [i])
    rho_j = reduced_density_matrix(psi, N, [j])
    rho_ij = reduced_density_matrix(psi, N, [i, j])
    S_i = von_neumann_entropy(rho_i)
    S_j = von_neumann_entropy(rho_j)
    S_ij = von_neumann_entropy(rho_ij)
    return S_i + S_j - S_ij

def mutual_information_vs_distance(psi, N):
    dists = []
    I_vals = []
    for i in range(N):
        for j in range(i+1, N):
            d = j - i
            Iij = mutual_information(psi, N, i, j)
            dists.append(d)
            I_vals.append(Iij)
    dists = np.array(dists)
    I_vals = np.array(I_vals)
    max_d = N-1
    d_unique = []
    I_mean = []
    for d in range(1, max_d+1):
        mask = dists == d
        if np.any(mask):
            d_unique.append(d)
            I_mean.append(np.mean(I_vals[mask]))
    return np.array(d_unique), np.array(I_mean)

def time_evolution_eig(eigvals, eigvecs, psi0, times):
    c0 = eigvecs.conj().T @ psi0
    states = []
    for t in times:
        phase = np.exp(-1j * eigvals * t)
        psi_t = eigvecs @ (phase * c0)
        states.append(psi_t)
    return states

def half_chain_entropy_vs_time(states, N):
    S_list = []
    half_sites = list(range(N//2))
    for psi in states:
        rho_half = reduced_density_matrix(psi, N, half_sites)
        S_half = von_neumann_entropy(rho_half)
        S_list.append(S_half)
    return np.array(S_list)

def mi_distance_vs_time(states, N):
    t_count = len(states)
    d_max = N-1
    d_vals = np.arange(1, d_max+1)
    I_dt = np.zeros((d_max, t_count), dtype=float)
    centers = [N//2 - 1, N//2] if N >= 4 else [N//2]
    for ti, psi in enumerate(states):
        counts = np.zeros(d_max, dtype=int)
        for i0 in centers:
            for j in range(N):
                if j == i0:
                    continue
                d = abs(j - i0)
                Iij = mutual_information(psi, N, i0, j)
                I_dt[d-1, ti] += Iij
                counts[d-1] += 1
        mask = counts > 0
        I_dt[mask, ti] /= counts[mask]
    return d_vals, I_dt


def block_entropies_ground_state(N=8, J=1.0, h=0.5):
    """
    Compute block entropies S(ell) [bits] for ell = 1..N//2
    in the ground state of the TFIM with parameters (N, J, h).
    The block is taken as the contiguous set of sites {0, ..., ell-1}.
    Returns (ells, S_bits, E0).
    """
    H = tfim_hamiltonian(N, J, h)
    E0, psi_gs, eigvals, eigvecs = ground_state(H)
    ells = []
    S_bits = []
    for ell in range(1, N // 2 + 1):
        subsystem = list(range(ell))
        rho_block = reduced_density_matrix(psi_gs, N, subsystem)
        S_nats = von_neumann_entropy(rho_block)
        S_bits.append(S_nats / np.log(2.0))
        ells.append(ell)
    return np.array(ells), np.array(S_bits), E0


def plot_block_entropies(N=8, J=1.0, h=0.5, fig_dir="figures", prefix="TFIM"):
    """
    Compute and plot block entropies S(ell) [bits] for the TFIM ground state.
    Saves a PNG figure and returns (ells, S_bits, E0).
    """
    from pathlib import Path
    ells, S_bits, E0 = block_entropies_ground_state(N=N, J=J, h=h)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(ells, S_bits, 'o-')
    plt.xlabel('Block length ell')
    plt.ylabel('Entanglement entropy S(ell) [bits]')
    plt.title(f'TFIM ground state, N={N}, J={J}, h={h}')
    plt.tight_layout()
    out = fig_dir / f"{prefix}_BlockEntropy_N{N}.png"
    plt.savefig(out)
    plt.close()
    return ells, S_bits, E0


def run_static(N=8, J=1.0, h=0.5, fig_dir="figures", prefix="TFIM"):
    """Compute ground-state I(d) and save a plot."""
    H = tfim_hamiltonian(N, J, h)
    E0, psi_gs, eigvals, eigvecs = ground_state(H)
    d, I_d_nats = mutual_information_vs_distance(psi_gs, N)
    I_d_bits = I_d_nats / np.log(2.0)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.semilogy(d, I_d_bits, 'o-')
    plt.xlabel('Distance d = |i-j|')
    plt.ylabel('Mutual information I(d) [bits]')
    plt.title(f'TFIM ground state, N={{N}}, J={{J}}, h={{h}}')
    plt.tight_layout()
    out = fig_dir / f"{prefix}_MI_distance_N{N}.png"
    plt.savefig(out)
    plt.close()
    return d, I_d_bits, E0

def run_dynamic(N=8, J=1.0, h=0.5, t_max=6.0, num_t=60, fig_dir="figures", prefix="TFIM"):
    """Unitary evolution from |↑...↑>, half-chain entropy and I(d,t)."""
    H = tfim_hamiltonian(N, J, h)
    E0, psi_gs, eigvals, eigvecs = ground_state(H)
    psi0 = basis_state_z(N, up=True)
    times = np.linspace(0.0, t_max, num_t)
    states_t = time_evolution_eig(eigvals, eigvecs, psi0, times)
    S_half_t_nats = half_chain_entropy_vs_time(states_t, N)
    S_half_t_bits = S_half_t_nats / np.log(2.0)
    d_vals, I_dt_nats = mi_distance_vs_time(states_t, N)
    I_dt_bits = I_dt_nats / np.log(2.0)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    # entropy plot
    plt.figure()
    plt.plot(times, S_half_t_bits)
    plt.xlabel('Time t')
    plt.ylabel('Half-chain entropy S_A(t) [bits]')
    plt.title(f'TFIM N={{N}}, J={{J}}, h={{h}}: half-chain entanglement')
    plt.tight_layout()
    out_ent = fig_dir / f"{prefix}_Tevo_HalfEntropy_N{N}.png"
    plt.savefig(out_ent)
    plt.close()
    # MI heatmap
    plt.figure()
    extent = (times[0], times[-1], d_vals[0], d_vals[-1])
    plt.imshow(I_dt_bits, aspect='auto', origin='lower', extent=extent)
    plt.xlabel('Time t')
    plt.ylabel('Distance d')
    plt.title(f'TFIM N={{N}}, J={{J}}, h={{h}}: I(d,t) [bits]')
    plt.colorbar(label='I(d,t) [bits]')
    plt.tight_layout()
    out_mi = fig_dir / f"{prefix}_Tevo_MI_Heatmap_N{N}.png"
    plt.savefig(out_mi)
    plt.close()
    return times, S_half_t_bits, d_vals, I_dt_bits

def run_demo(mode="both", **kwargs):
    if mode in ("static", "both"):
        run_static(**kwargs)
    if mode in ("dynamic", "both"):
        run_dynamic(**kwargs)
