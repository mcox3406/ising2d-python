from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from numba import njit
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

TC_ONSAGER = 2 / np.log(1 + np.sqrt(2))

@njit
def init_lattice(N, lattice_type=0):
    """
    Initialize the lattice with spins.

    lattice_type (int):
        0 - Random lattice
        1 - Checkerboard pattern
        2 - Uniform split
    """
    if lattice_type == 1:
        # checkerboard pattern
        A = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            for j in range(N):
                A[i, j] = 1 if (i + j) % 2 == 0 else -1
    elif lattice_type == 2:
        # uniform split
        A = np.ones((N, N), dtype=np.int8)
        A[:, N//2:] = -1
    else:
        # random lattice
        A = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(N, N))
    return A

@njit
def ising_mc_step(N, A, T, B):
    """
    Perform a single Monte Carlo step using the Metropolis algorithm.
    """
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    s = A[i, j]

    # periodic boundary conditions
    ileft = (i - 1) % N
    iright = (i + 1) % N
    jdown = (j - 1) % N
    jup = (j + 1) % N

    neighbor_sum = A[ileft, j] + A[iright, j] + A[i, jdown] + A[i, jup]
    delta_E = 2 * s * (neighbor_sum + B)
    if delta_E <= 0:
        A[i, j] = -s
    else:
        if np.random.rand() < np.exp(-delta_E / T):
            A[i, j] = -s
    return A

@njit
def ising_calc_energy(N, A, B):
    """
    Calculate the total energy of the lattice.
    """
    E = 0.0
    for i in range(N):
        for j in range(N):
            s = A[i, j]
            s_right = A[i, (j + 1) % N]
            s_down = A[(i + 1) % N, j]
            E -= s * (s_right + s_down)
    E -= B * np.sum(A)
    return E

@njit
def ising_calc_magnetization(A):
    """
    Calculate the total magnetization of the lattice.
    """
    return np.sum(A)

def ising_run(N, B, A, T, Nsteps, samp_freq):
    """
    Run the Ising model simulation for a given temperature.
    """
    samples = Nsteps // samp_freq + 1
    e_samples = np.zeros(samples)
    m_samples = np.zeros(samples)
    e_samples[0] = ising_calc_energy(N, A, B)
    m_samples[0] = ising_calc_magnetization(A)
    for step in range(1, Nsteps + 1):
        A = ising_mc_step(N, A, T, B)
        if step % samp_freq == 0:
            idx = step // samp_freq
            e_samples[idx] = ising_calc_energy(N, A, B)
            m_samples[idx] = ising_calc_magnetization(A)
    return e_samples, m_samples, A


def ising2d(T_list, N, lattice_type=0, B=0, eqsteps=10**5, sfreq_eq=10**4, 
            prodsteps=5*10**6, sfreq_prod=10**4, save_data=False, data_dir='data', plot_lattice=False,
            figs_dir='figs', lattice_filename='lattices', reinitialize_lattice=False):
    """
    Run the 2D Ising model simulation over a range of temperatures.
    """
    T_list = np.sort(T_list)[::-1]
    num_samples = prodsteps // sfreq_prod + 1
    E = np.zeros((len(T_list), num_samples))
    M = np.zeros((len(T_list), num_samples))
    A = init_lattice(N, lattice_type)

    if save_data and not os.path.exists(data_dir):
        os.makedirs(data_dir)

    final_lattices = []
    final_T = []

    for i, T in tqdm(enumerate(T_list), total=len(T_list)):
        # logging.info(f'Simulating T = {T:.3f}')

        if reinitialize_lattice:
            A = init_lattice(N, lattice_type)

        # equilibration
        _, _, A = ising_run(N, B, A, T, eqsteps, sfreq_eq)

        # production run
        e_samples, m_samples, A = ising_run(N, B, A, T, prodsteps, sfreq_prod)
        E[i, :] = e_samples
        M[i, :] = m_samples

        if save_data:
            np.save(os.path.join(data_dir, f'energies_T{T:.3f}.npy'), e_samples)
            np.save(os.path.join(data_dir, f'magnetizations_T{T:.3f}.npy'), m_samples)

        final_lattices.append(A.copy())
        final_T.append(T)
        
    if plot_lattice and final_lattices:
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)

        num_plots = len(final_lattices)
        cols = 5
        rows = math.ceil(num_plots / cols)
        fig = plt.figure(figsize=(3*cols, 3*rows))
        gs = GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.4)

        for idx, (A_final, T_final) in enumerate(zip(final_lattices, final_T)):
            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(A_final, cmap='gray', interpolation='nearest')
            ax.set_title(f'T = {T_final:.2f}')
            ax.axis('off')

        # plt.suptitle('Final Lattice States at Different Dimensionless Temperatures', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(figs_dir, f'{lattice_filename}.pdf'), bbox_inches='tight')
        plt.show()

    return E, M, T_list