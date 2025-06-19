"""Functions for vectorized Lindbladian and Davies map
   NOTE: the Davies map is constructed in the eigenbasis of the system Hamiltonian. Transform back to computational basis if needed.
"""
import numpy as np
import scipy.linalg
from typing import Tuple


def vectorized_lindbladian(n_sites: int, lind_op_list: list, hamiltonian: np.ndarray) -> np.ndarray:
    """Given a system with n_sites, a hamiltonian and a list of jump operators, returns the vectorized lindbladian

    Parameters
    ----------
    n_sites : int
        number of sites
    lind_op_list : list
        list of jump operators
    hamiltonian : np.ndarray
        hamiltonian

    Returns
    -------
    np.ndarray
        matrix representing the Liouvillian generator in the vectorized form
    """
    vectorized_lindbladian = -1j*np.kron(hamiltonian, np.eye(2**n_sites)) +1j*np.kron(np.eye(2**n_sites), np.transpose(hamiltonian))
    for jump_op in lind_op_list:
        vectorized_lindbladian += np.kron(jump_op, np.conjugate(jump_op) ) - 0.5*np.kron(np.conjugate(np.transpose(jump_op)) @ jump_op, np.eye(2**n_sites) ) - 0.5*np.kron(np.eye(2**n_sites), np.transpose(jump_op) @ np.conjugate(jump_op))
    return vectorized_lindbladian


def fermi_function(beta: float, energy: float) -> float:
    """Fermi function for a given inverse temperature and energy.
    Parameters
    ----------
    energy : float
        energy
    beta : float
        inverse temperature

    Returns
    -------
    float
        Fermi Function
    """
    return 1./(np.exp(beta*energy)+1)


def jump_operators_for_davies_map(n_sites: int, hamiltonian: np.ndarray, beta: float) -> list:
    """List of jump operators that define the FERMIONIC Davies map in the eigenbasis of the system Hamiltonian.
       The fixed point of the Lindblad dynamics is the thermal state.

    Parameters
    ----------
    n_sites : int
        number of sites
    hamiltonian : np.ndarray
        hamiltonian
    beta : float
        inverse temperature

    Returns
    -------
    list
        list of jump operators that define the Davies dissipator
    """
    eigw_hamiltonian = np.linalg.eigh(hamiltonian)[0] 
    hamiltonian_diag_basis = np.eye(2**n_sites, dtype=np.complex128)
    lind_op_list = []
    for i in range(2**n_sites):
        for j in range(2**n_sites):
            lind_op_list.append( np.sqrt(1-fermi_function(beta, eigw_hamiltonian[j]-eigw_hamiltonian[i] ) )* np.outer( hamiltonian_diag_basis[:,i], np.conjugate(hamiltonian_diag_basis[:,j]) ) )
            lind_op_list.append( np.sqrt(fermi_function(beta, eigw_hamiltonian[j]-eigw_hamiltonian[i] ) )* np.outer( hamiltonian_diag_basis[:,j], np.conjugate(hamiltonian_diag_basis[:,i]) ) )
    return lind_op_list

def bose_function(beta: float, energy: float) -> float:
    """Bose-Einstein function for a given inverse temperature and energy.
    Parameters
    ----------
    energy : float
        energy
    beta : float
        inverse temperature

    Returns
    -------
    float
        Fermi Function
    """
    return 1./(np.exp(energy*beta)-1.)


def jump_operators_for_davies_map_spin_onehalf(n_sites: int, hamiltonian: np.ndarray, beta: float) -> list:
    """List of jump operators that define the Davies map in the eigenbasis of the system Hamiltonian.
       The fixed point of the Lindblad dynamics is the thermal state.

    Parameters
    ----------
    n_sites : int
        number of sites
    hamiltonian : np.ndarray
        hamiltonian
    beta : float
        inverse temperature

    Returns
    -------
    list
        list of jump operators that define the Davies dissipator
    """
    import cmath
    eigw_hamiltonian = np.linalg.eigh(hamiltonian)[0] 
    hamiltonian_diag_basis = np.eye(2**n_sites, dtype=np.complex128)
    
    lind_op_list = [] 

    for i in range(2**n_sites):
        for j in range(i):
            lind_op_list.append( cmath.sqrt( bose_function(beta, eigw_hamiltonian[j]-eigw_hamiltonian[i]) +1 )* np.outer( hamiltonian_diag_basis[:,i], np.conjugate(hamiltonian_diag_basis[:,j]) ) )
            lind_op_list.append( cmath.sqrt( bose_function(beta, eigw_hamiltonian[j]-eigw_hamiltonian[i]) )* np.outer( hamiltonian_diag_basis[:,j], np.conjugate(hamiltonian_diag_basis[:,i]) ) )

    return lind_op_list



def spectrum_of_vectorized_generator(
    lindbladian_vectorized: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the spectrum (eigenvalues and eigenvectors) of a vectorized
    generator (e.g., a Lindbladian superoperator).

    The eigenvalues are sorted by the absolute value of their real parts.

    Parameters:
        lindbladian_vectorized (np.ndarray): Square matrix representing the 
                                              vectorized generator.

    Returns:
        Tuple containing:
            - spectrum (np.ndarray): Array of shape (N, 2) with real and imaginary 
                                     parts of eigenvalues. Columns: [Re(λ), Im(λ)].
            - eigw_vectorized_lindbladian (np.array): Array of shape (N) containing also the ordered eigenvalues.
            - left_eigenvectors (np.ndarray): Matrix whose columns are left eigenvectors.
            - right_eigenvectors (np.ndarray): Matrix whose columns are right eigenvectors.
    """
    eigw_vectorized_lindbladian, left_eigv_vectorized_lindbladian, right_eigv_vectorized_lindbladian = scipy.linalg.eig(lindbladian_vectorized, left=True, right=True)
    idx = np.argsort(np.abs(np.real(eigw_vectorized_lindbladian)))
    eigw_vectorized_lindbladian = eigw_vectorized_lindbladian[idx]
    left_eigv_vectorized_lindbladian = left_eigv_vectorized_lindbladian[:,idx]
    right_eigv_vectorized_lindbladian = right_eigv_vectorized_lindbladian[:,idx]
    spectrum = np.column_stack((np.real(eigw_vectorized_lindbladian), np.imag(eigw_vectorized_lindbladian))) 
    
    return spectrum, eigw_vectorized_lindbladian, left_eigv_vectorized_lindbladian, right_eigv_vectorized_lindbladian