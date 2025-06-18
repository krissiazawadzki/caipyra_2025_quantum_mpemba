import numpy as np

def Svn(rho: np.array):
    """
    Calculates the von Neumann entropy of state $\rho$.

        $S(\rho) = - \Tr{\rho \log(\rho)}$

    Inputs:
        - rho: quantum state (np.array, matrix)
    
    Outputs:
        - S: entropy of state (float)
    
    """
    lambda_j = np.linalg.eigvals(rho)
    return - (lambda_j * np.log(lambda_j)).sum()

'''


'''
def relative_entropy_Kullback(rho: np.array, sigma: np.array):
    """
    Function computes the relative distance between two states $\rho$ and $\sigma$ 
    using the Kullback relative entropy.

    Inputs:
        - rho: density matrix (np.array, matrix)
        - sigma: density matrix (np.array, matrix)

    Outputs:
        - D: trace distance between the two states (float)
    
    """
    p_i, V_i = np.linalg.eigh(rho)
    q_i, W_i = np.linalg.eigh(sigma)
    Vi_Wj = V_i.T.conjugate() @ W_i
    Vi_Wj *= Vi_Wj.conjugate()
    D = p_i @ np.log(p_i) - p_i @ Vi_Wj @ np.log(q_i)  
    return D



def dephased_state(rho: np.array, m_f_l: np.array):
    """
    Dephases a density matrix $\rho$ in the eigenbases $\{ m_f \}$ of an operator 
    (the Hamiltonian, for example).

    Inputs:
        - rho: the density matrix (np.array, matrix)
        - m_f_l: the eigenbasis to dephase rho. Eigenvectors are organized in columns (np.array, matrix) 
        
    Outputs:
        - D_f: dephased state (np.array, matrix)
    
    """
    dim = m_f_l.shape[0]
    D_f = np.zeros_like(rho)
    for m in range(dim):
        mf = m_f_l[:,m].reshape(dim,1)
        D_f += (mf.T.conjugate() @ rho @ mf) * mf @ mf.T.conjugate()
    return D_f

def relative_entropy_coherence(rho: np.array, m_f: np.array, U: np.array=None):
    """
    Computes the relative entropy of coherence of a state with 
    respect to an eigenbasis of another operator. 

    Inputs:
        - rho: the density matrix (np.array, matrix)
        - m_f_l: the eigenbasis to dephase rho. Eigenvectors are organized in columns (np.array, matrix) 
        - U: unitary matrix to rotate the basis m_f (np.array, matrix)

    Outputs:
        - REC: relative entropy of coherence (float)    
    
    """
    if U is None:
        U = np.eye(m_f.shape[0])
    m_f_l = U.T.conjugate() @ m_f
    D_f = dephased_state(rho, m_f_l)
    REC = Svn(D_f) - Svn(rho)
    return REC



def coherence(rho: np.array, vE_m: np.array, diffS_or_KullbackLieb: str ='diffS'):
    """
    Computes the coherence of a state rho with respect 
    to the energy eigenbasis vE_m. This calculation can
    use the (i) difference in the von-Neuman entropy of rho 
    and rho dephased in the basis vE_m or or (ii) the 
    Kullback-Lieb divergence. 

    Inputs:
        - rho: the density matrix (np.array, matrix)
        - vE_m: the eigenbasis of the Hamiltonian. Eigenvectors 
        are organized in columns (np.array, matrix) 
        - diffS_or_KullbackLieb: string to indicate if the 
        calculation will use
        'diffS': the difference in entropies
        S(D_H[rho]) - S(rho)
        or 
        'KullbackLieb': the Kullback-Lieb distance 
        S(D_H[rho]||rho)

    Outputs:
        - C(rho): coherence (float)
    
    """
    if(diffS_or_KullbackLieb == 'diffS'):
        return relative_entropy_coherence(rho, vE_m)
    if(diffS_or_KullbackLieb == 'KullbackLieb'):
        return relative_entropy_Kullback(rho, dephased_state(rho, vE_m))
    else: 
        raise ValueError("'{} equation' not recognized, must be 'diffS' or 'KullbackLieb'")
    

def entropy_flux(rho_ts, ts: np.array, E_n: np.array, vE_n : np.array, beta: float):
    """
    Computes the entropy flux defined as 
    \phi = -\frac{1}{T} \sum_n E_n \frac{d p_n}{dt}

    where 
    p_n are the populations of the instantaneous density 
    matrix rho_t in the energy eigenbasis
    E_n are the energies of the system
    vE_n are the eigenvectors of the system

    Inputs:
        - rho_ts: array with instantaneous the density matrix (np.array, matrix)
        - ts: time at which rho_ts where evaluates  (np.array)
        - E_n: energies (np.array)
        - vE_m: the eigenbasis of the Hamiltonian. Eigenvectors 
        are organized in columns (np.array, matrix) 
        - beta: inverse temperature (float)

    Ouputs:
    - flux: the value of phi    
    
    """

    p_n_ts = []
    for i, t in enumerate(ts):
        rho_t = rho_ts[i]
        p_n = np.diag(vE_n.T.conjugate() @ rho_t @ vE_n) 
        p_n_ts.append(p_n)

    p_n_ts = np.array(p_n_ts)
    dp_n_dts = np.diff(p_n_ts, axis=0) / np.diff(ts)

    flux = - beta * np.einsum("n, tn->t", E_n, dp_n_dts) 
    return flux



def boltzman_weights(E_n: np.array, beta: float):
    """
    Calculates the Boltzman weights p_n = e^{-beta E_n} / Z, where Z = \sum_n e^{-beta E_n}

    Inputs:
        - E_n: eigenenergies of Hamiltonian, np.array
        - beta: inverse temperature, float
	
	Outputs: 
        - p_n: Boltzman weights (not normalized by the partition function)    
    """
    p_n = np.exp(-(E_n-E_n[0])*beta) / (1.0 + np.exp(-(E_n[1:]-E_n[0])*beta).sum())
    return p_n





def Gibbs_state(E_n: np.array, Psi_n: np.array, beta: float, return_Z=True):
    """
    Calculates the Gibbs state for a system defined by a Hamiltonian $H$ at inverse 
    temperature $\beta = T^{-1}$. 

        $\rho_{Gibbs} = \exp{-\beta H} / Z$

        with $Z = \Tr{\exp{-\beta H}}$.

    In the eigenbasis $\{E_n, \Psi_n\}$ of the Hamiltonian $H$, this state corresponds to
        $\rho_{Gibbs} = \sum_n p_n \ket{\Psi_n} \bra{\Psi_m}$,

    where $p_n = \exp{-\beta E_n} / Z$ are the Boltzman weights.

    Inputs:
        - E_n: eigen-energies of a Hamiltonian (np.array)
        - Psi_n: eigen-states of a Hamiltonian (np.array: matrix)
        - beta: inverse temperature (float)
        - return_Z: flag to return or not the partition function (bool)
	
	Outputs: 
        - rho_Gibbs: density matrix (np.array)
        - Z: partition function (float)    

    """
    rho_Gibbs = np.zeros_like(Psi_n)
    p_n = boltzman_weights(E_n, beta)
    for n, p_n in enumerate(p_n):
        rho_Gibbs += p_n[n] * np.outer(Psi_n[:,n], Psi_n[:,n].conjugate())
    
    Z = p_n.sum()

    rho_Gibbs /= Z
    
    if(return_Z is False):
        return rho_Gibbs

    return rho_Gibbs, Z