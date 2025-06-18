import numpy as np
from entropy_and_coherences import Svn
import scipy

def energy(rho: np.ndarray, hamiltonian: np.ndarray) -> float:
    """
    Computes the average energy of a quantum state.

    Parameters:
        rho (np.ndarray): Density matrix.
        hamiltonian (np.ndarray): Hamiltonian matrix.

    Returns:
        float: Average energy.
    """
    return float(np.trace(rho @ hamiltonian).real)

def noneq_free_energy(rho: np.ndarray, hamiltonian: np.ndarray, temperature_bath: float) -> float:
    """
    Computes the non-equilibrium free energy of a quantum state.

    Formula:
        F = E + T * S

    where
        E = Tr[rho * H]
        S = Tr[rho * log(rho)]

    Parameters:
        rho (np.ndarray): Density matrix.
        hamiltonian (np.ndarray): Hamiltonian matrix.
        temperature_bath (float): Temperature (in units where k_B = 1).

    Returns:
        float: Non-equilibrium free energy.
    """
    E = energy(rho, hamiltonian)
    S = np.trace(rho @ scipy.linalg.logm(rho))
    return E + temperature_bath * S