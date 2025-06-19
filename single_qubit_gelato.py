import matplotlib.pyplot as plt
import numpy as np
import os, sys
import scipy
import argparse
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

sys.path.append('src/qtd/') 
import vectorized_lindbladian_and_davies_map as vec_lind
import entropy_and_coherences as ent_and_coh
from energies import energy, noneq_free_energy

sys.path.append('src/qubit_ops/')
from qubit_operators import sz_op, bloch_representation_to_computational, computational_to_bloch_representation

n_sites = 1

# Argument parser
parser = argparse.ArgumentParser(description="Simulation parameters")

# Working parameters
parser.add_argument('--seed', type=int, default=0,help='Random seed (default: 0)')
parser.add_argument('--temperature_bath', type=float, default=10,help='Bath temperature (default: 10)')

# Time parameters
parser.add_argument('--t_max', type=float, default=8,help='Maximum simulation time (default: 8)')
parser.add_argument('--dt', type=float, default=0.01,help='Time step (default: 0.01)')

# Qubit parameters
parser.add_argument('--omega', type=float, default=5,help='Qubit frequency (default: 5)')

# Initial Bloch vectors
parser.add_argument('--initial_state', type=float, nargs=3, default=[0.276, 0.359, 0.303],help='Initial state Bloch vectors (e.g., rx ry rz)')

# Flag to request random state
parser.add_argument('--random_initial_state', action='store_true',help='If set, use a random initial state (overrides --initial_state)')


# Data directory
parser.add_argument('--dir_data', type=str, default='data/',help='Directory to save data (default: data/)')

# Parse arguments
args = parser.parse_args()

# Use arguments
seed = args.seed
temperature_bath = args.temperature_bath
t_max = args.t_max
dt = args.dt
omega = args.omega
dir_data = args.dir_data

# Create time vector
time_v = np.arange(0, t_max + dt, dt)
Nt = len(time_v)

# Create data directory if it doesn't exist
os.makedirs(dir_data, exist_ok=True)

# random seed 
np.random.seed(seed)

# Hamiltonian prop. to \sigma_z
ham = omega*sz_op

# get Hamiltonian eigenvalues
eigw_hamiltonian = np.linalg.eigh(ham)[0]

# Davies map
jump_op_list = vec_lind.jump_operators_for_davies_map_spin_onehalf(n_sites, ham, 1./temperature_bath)
lindbladian_vectorized = vec_lind.vectorized_lindbladian(n_sites, jump_op_list, np.diag(eigw_hamiltonian) )

# spectrum of the generator
spectrum, eigw_vectorized_lindbladian, left_eigv_vectorized_lindbladian, right_eigv_vectorized_lindbladian = vec_lind.spectrum_of_vectorized_generator(lindbladian_vectorized)

np.savetxt(f'{dir_data}/spectrum.txt', spectrum, fmt='%.15f', delimiter='\t', comments='')

if args.random_initial_state:
    r_vector = np.random.rand(3)
    r_vector = r_vector / np.linalg.norm(r_vector)
    # multiply r_vector by a diagonalized number between 0 (maximally mixed) and 1 (pure)
    r_vector = r_vector * np.random.rand(1)
else:
    r_vector = np.array(args.initial_state)

print('r_vector = ', r_vector)

# compute density matrix from r_vector
state_original = bloch_representation_to_computational(r_vector)
print("state_original = \n ", state_original)

#diagonalize original state
eigw_original_state = np.linalg.eigh(state_original)[0]
state_diagonalized = np.diag(eigw_original_state)
print("state_diagonalized = \n", state_diagonalized)
state_diagonalized_bloch = computational_to_bloch_representation(state_diagonalized)
print("state_diagonalized_bloch = ", state_diagonalized_bloch)



F_state_original = noneq_free_energy(state_original, np.diag(eigw_hamiltonian), temperature_bath)
F_state_diagonalized = noneq_free_energy(state_diagonalized, np.diag(eigw_hamiltonian), temperature_bath)
print("F_state_original = ", F_state_original)
print("F_state_diagonalized = ", F_state_diagonalized)

#Time evolution
state_original_vectorized_every_t = np.zeros((4**n_sites,int(t_max/dt)+1), dtype=np.complex128)
state_diagonalized_vectorized_every_t = np.zeros((4**n_sites,int(t_max/dt)+1), dtype=np.complex128)
distance_original = []
distance_diagonalized = []
F_original = []
F_diagonalized = []
energy_original = []
energy_diagonalized = []
entropy_original = []
entropy_diagonalized = []
classical_relative_entropy_original = []
classical_relative_entropy_diagonalized = []
quantum_relative_entropy_original = []
quantum_relative_entropy_diagonalized = []

original_state_original_vectorized = state_original.flatten()
original_state_diagonalized_vectorized = state_diagonalized.flatten()
id_v = np.eye(2**n_sites).flatten()
steady_state = np.reshape(right_eigv_vectorized_lindbladian[:,0], (2**n_sites,2**n_sites))
steady_state /= np.trace(steady_state)
steady_state_bloch = computational_to_bloch_representation(steady_state)

for t in np.arange(0, t_max+dt, dt):
    #tevo
    state_original_vectorized = scipy.linalg.expm(t*lindbladian_vectorized) @ original_state_original_vectorized
    state_diagonalized_vectorized = scipy.linalg.expm(t*lindbladian_vectorized) @ original_state_diagonalized_vectorized
    state_original = np.reshape(state_original_vectorized, (2**n_sites,2**n_sites))
    state_diagonalized = np.reshape(state_diagonalized_vectorized, (2**n_sites,2**n_sites))
    state_original /= np.trace(state_original)
    state_diagonalized /= np.trace(state_diagonalized)
    
    #state at every timestep
    state_original_vectorized_every_t[:,int(t/dt)] = state_original_vectorized
    state_diagonalized_vectorized_every_t[:,int(t/dt)] = state_diagonalized_vectorized
    #L1 distance from steady state
    distance_original.append( np.sum(np.abs(state_original - steady_state)) )
    distance_diagonalized.append( np.sum(np.abs(state_diagonalized - steady_state)) )
    #non-equilibrium free energy
    F_original.append( noneq_free_energy(state_original, np.diag(eigw_hamiltonian), temperature_bath) )
    F_diagonalized.append( noneq_free_energy(state_diagonalized, np.diag(eigw_hamiltonian), temperature_bath) )
    #energy
    energy_original.append( energy(state_original, np.diag(eigw_hamiltonian)) )
    energy_diagonalized.append( energy(state_diagonalized, np.diag(eigw_hamiltonian)) )
    #entropy
    entropy_original.append( ent_and_coh.Svn(state_original) )
    entropy_diagonalized.append( ent_and_coh.Svn(state_diagonalized) )
    #classical relative entropy
    classical_relative_entropy_original.append( ent_and_coh.relative_entropy_Kullback( state_original, steady_state))
    classical_relative_entropy_diagonalized.append( ent_and_coh.relative_entropy_Kullback( state_diagonalized, steady_state))
    #quantum rel. entr.
    quantum_relative_entropy_original.append( ent_and_coh.relative_entropy_coherence( state_original, np.eye((2**n_sites), dtype='complex' ) ) )
    quantum_relative_entropy_diagonalized.append( ent_and_coh.relative_entropy_coherence( state_diagonalized, np.eye((2**n_sites), dtype='complex' ) ) )
    
    
#compute time derivative of the free energy
entropy_production_original = []
entropy_production_diagonalized = []
for t in range(1,int(t_max/dt)+1):
    entropy_production_original.append ( -1./temperature_bath*(F_original[t] - F_original[t-1])/dt )
    entropy_production_diagonalized.append ( -1./temperature_bath*(F_diagonalized[t] - F_diagonalized[t-1])/dt )
    
#transform the time evolved states from comptuational to bloch representation
state_original_every_t_bloch = np.zeros((3,int(t_max/dt)+1), dtype=np.complex128)
state_diagonalized_every_t_bloch = np.zeros((3,int(t_max/dt)+1), dtype=np.complex128)
for t in np.arange(0, t_max+dt, dt):
    state_original_every_t_bloch[:,int(t/dt)] = computational_to_bloch_representation( np.reshape( state_original_vectorized_every_t[:,int(t/dt)], (2,2) ) )
    state_diagonalized_every_t_bloch[:,int(t/dt)] = computational_to_bloch_representation( np.reshape( state_diagonalized_vectorized_every_t[:,int(t/dt)], (2,2) ) )

###PLOTS
color_original = '#66a61e' #'#beaed4'
color_diagonalization = '#7570b3' #'#fdc086'

# #plot only the noneq. free energy for the original and diagonalized state only up to T=2
fig, axs = plt.subplots(1)
axs.plot(time_v[:150], F_original[:150], label='original state', color=color_original)
axs.plot(time_v[:150], F_diagonalized[:150], label='diagonalized state', color=color_diagonalization)
axs.set_ylabel('Noneq. free energy')
axs.set_xlabel('time (1/J)')


axins = inset_axes(axs, width="65%", height="85%",bbox_to_anchor=(.3, .3, .6, .5), bbox_transform=axs.transAxes, loc='upper right')
axins.semilogy(time_v[:], distance_original[:], label='original state', color=color_original)
axins.semilogy(time_v[:], distance_diagonalized[:], label='diagonalized state', color=color_diagonalization)
#fit with lambda2 and lambda4
lambda_2 = distance_original[-1]*np.exp(spectrum[1,0]*(time_v[:]-time_v[-1]))
lambda_4 = distance_diagonalized[-1]*np.exp(spectrum[3,0]*(time_v[:]-time_v[-1]))
axins.semilogy(time_v[:], lambda_2, label='$\exp(\Re{(\lambda_2)}t) $', linestyle='dashed', color = color_original)
axins.semilogy(time_v[:], lambda_4, label='$\exp(\Re{(\lambda_3)}t) = \exp(\Re{(\lambda_4)}t) $', linestyle='dashed', color = color_diagonalization)

axins.set_ylabel('$L_1$ distance')
axins.set_xlabel('time (1/J)')
axins.legend(frameon=False)

axs.legend()



#PLOT TOTAL, QUANTUM AND CLASSICAL RELATIVE ENTROPY
fig, axs = plt.subplots(1)
axs.semilogy(time_v[:150], np.array(classical_relative_entropy_original[:150])+np.array(quantum_relative_entropy_original[:150]), label='total, original',color=color_original)
axs.semilogy(time_v[:150], classical_relative_entropy_original[:150], label='classical,original', color=color_original, linestyle='dashed')
axs.semilogy(time_v[:150], quantum_relative_entropy_original[:150], label='quantum, original', color=color_original, linestyle='dotted')

axs.semilogy(time_v[:150], np.array(classical_relative_entropy_diagonalized[:150])+np.array(quantum_relative_entropy_diagonalized[:150]), label='total, diagonalized', color=color_diagonalization)
axs.semilogy(time_v[:150], classical_relative_entropy_diagonalized[:150], label='classical, diagonalized', color=color_diagonalization, linestyle='dashed')

#fit total entropy production with lambda2 and lambda4
# lambda_2 = (np.array(classical_relative_entropy_original[80])+np.array(quantum_relative_entropy_original[80]))*np.exp(2*np.real(eigw_vectorized_lindbladian[1])*(time_v[:80]-time_v[80]))
# lambda_4 = (np.array(classical_relative_entropy_diagonalized[80])+np.array(quantum_relative_entropy_diagonalized[80]))*np.exp(2*np.real(eigw_vectorized_lindbladian[3])*(time_v[:80]-time_v[80]))
# axs.semilogy(time_v[:80], lambda_2, label='$\exp(\Re{(\lambda_2)}t) $', linestyle='dashed', color = 'red')
# axs.semilogy(time_v[:80], lambda_4, label='$\exp(\Re{(\lambda_3)}t) = \exp(\Re{(\lambda_4)}t) $', linestyle='dashed', color = 'orange')

plt.text(.5, 0.0019, r'$e^{-2\mathrm{Re}(\lambda_2)} = e^{-2\mathrm{Re}(\lambda_3)}$', rotation=-18, va='bottom', ha='left', color=color_original, fontsize=18)
plt.text(.5, 0.00007, r'$e^{-2\mathrm{Re}(\lambda_4)}$', rotation=-28, va='bottom', ha='left', color=color_diagonalization, fontsize=18)

# axs.semilogy(time_v[1:], entropy_production_original, label='entropy production original', color=color_original)
# axs.semilogy(time_v[1:], entropy_production_diagonalized, label='entropy production diagonalized', color=color_diagonalization)



axs.set_ylabel('$\Pi(t)$')
axs.set_xlabel('time $(1/J)$')
axs.legend()
plt.show()


#save noneq. free energy and L1 distance and L1 fit for plot
# spectrum = np.column_stack((np.real(eigw_vectorized_lindbladian), np.imag(eigw_vectorized_lindbladian))) 
# 
np.savetxt(f'{dir_data}/spectrum.txt', spectrum, fmt='%.15f', delimiter='\t', comments='')
F_original = np.column_stack((time_v[:150], np.real(F_original)[:150] ) )
F_diagonalized = np.column_stack((time_v[:150], np.real(F_diagonalized)[:150] ) )

L1_original = np.column_stack((time_v[:], np.real(distance_original)[:] ) )
L1_diagonalized = np.column_stack((time_v[:], np.real(distance_diagonalized)[:] ) )
lambda_2_fit_original = np.column_stack((time_v[:], np.real(lambda_2) ) )
lambda_4_fit_diagonalized = np.column_stack((time_v[:], np.real(lambda_4) ) )


np.savetxt(f'{dir_data}/F_original.txt', F_original, fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/F_diagonalized.txt', F_diagonalized, fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/L1_distance_original.txt', L1_original, fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/L1_distance_diagonalized.txt', L1_diagonalized, fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/lambda_2_fit_original.txt', lambda_2_fit_original, fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/lambda_4_fit_diagonalized.txt', lambda_4_fit_diagonalized, fmt='%.15f', delimiter='\t', comments='')

#save Bloch vectors of original, diagonalized and thermal steady state

np.savetxt(f'{dir_data}/bloch_original.txt', r_vector, fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/bloch_vector_diagonalized.txt', np.real(state_diagonalized_bloch),  fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/bloch_steady_state.txt', np.real(steady_state_bloch),  fmt='%.15f', delimiter='\t', comments='')

#save state dynamics in Bloch sphere
state_original_every_t_bloch_stack = np.column_stack((time_v[:150], np.real(state_original_every_t_bloch)[0,:150], np.real(state_original_every_t_bloch)[1,:150], np.real(state_original_every_t_bloch)[2,:150] ) )
state_diagonalized_every_t_bloch_stack = np.column_stack((time_v[:150], np.real(state_diagonalized_every_t_bloch)[0,:150], np.real(state_diagonalized_every_t_bloch)[1,:150], np.real(state_diagonalized_every_t_bloch)[2,:150] ) )
np.savetxt(f"{dir_data}/state_original_every_t_bloch.txt",state_original_every_t_bloch_stack)
np.savetxt(f"{dir_data}/state_diagonalized_every_t_bloch.txt",state_diagonalized_every_t_bloch_stack)

#save the entropy production
total_relative_entropy_original_stack = np.column_stack((time_v[:150], np.real( np.array(classical_relative_entropy_original[:150]) + np.array(quantum_relative_entropy_original[:150]) ) ) )
classical_relative_entropy_original_stack = np.column_stack((time_v[:150], np.real(np.array(classical_relative_entropy_original[:150])  )) )
quantum_relative_entropy_original_stack = np.column_stack((time_v[:150], np.real(np.array(quantum_relative_entropy_original[:150]))  ) )
total_relative_entropy_diagonalized_stack = np.column_stack( (time_v[:150], np.real(np.array(classical_relative_entropy_diagonalized[:150])+np.array(quantum_relative_entropy_diagonalized[:150]))) )
classical_relative_entropy_diagonalized_stack = np.column_stack( (time_v[:150], np.real(np.array(classical_relative_entropy_diagonalized[:150]))) )

np.savetxt(f'{dir_data}/total_relative_entropy_original.txt', total_relative_entropy_original_stack,  fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/classical_relative_entropy_original.txt', classical_relative_entropy_original_stack,  fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/quantum_relative_entropy_original.txt', quantum_relative_entropy_original_stack,  fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/total_relative_entropy_diagonalized.txt', total_relative_entropy_diagonalized_stack,  fmt='%.15f', delimiter='\t', comments='')

np.savetxt(f'{dir_data}/classical_relative_entropy_diagonalized.txt', classical_relative_entropy_diagonalized_stack,  fmt='%.15f', delimiter='\t', comments='')

