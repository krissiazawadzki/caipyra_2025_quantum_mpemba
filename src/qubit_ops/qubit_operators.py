import numpy as np
from typing import List

# Pauli operators
sx_op = np.array([[0, 1], [1, 0]], dtype=complex)
sy_op = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz_op = np.array([[1, 0], [0, -1]], dtype=complex)

def q_spherical_to_cartesian(r: float, theta: float, phi: float) -> List[float]:
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters:
        r (float): Radius (0 <= r).
        theta (float): Polar angle in radians (0 <= theta <= pi).
        phi (float): Azimuthal angle in radians (0 <= phi < 2*pi).

    Returns:
        list[float]: [x, y, z] Cartesian coordinates.
    """
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return [x, y, z]


def bloch_representation_to_computational(bloch_repr: np.ndarray) -> np.ndarray:
    """
    Converts a qubit Bloch vector representation into a density matrix.

    Parameters:
        bloch_repr (np.ndarray): A 3-element array representing the Bloch vector (rx, ry, rz).

    Returns:
        np.ndarray: 2x2 density matrix in the computational basis.
    """
    rx, ry, rz = bloch_repr

    # Bloch sphere representation: rho = 1/2 (I + rx*X + ry*Y + rz*Z)
    state = 0.5 * np.eye(2, dtype=complex) + rx * sx_op + ry * sy_op + rz * sz_op 

    return state


def computational_to_bloch_representation(state: np.ndarray) -> np.ndarray:
    """
    Converts a density matrix into its Bloch sphere representation.

    Parameters:
        state (np.ndarray): 2x2 density matrix in the computational basis.

    Returns:
        np.ndarray: 3-element array with Bloch vector components (rx, ry, rz).
    """
    # Compute expectation values of Pauli operators
    rx = 2 * np.trace(sx_op @ state.copy())
    ry = 2 * np.trace(sy_op @ state.copy())
    rz = 2 * np.trace(sz_op @ state.copy())

    # Return only the real part (imaginary parts should be zero up to numerical error)
    return np.real(np.array([rx, ry, rz]))