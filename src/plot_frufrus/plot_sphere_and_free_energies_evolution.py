import numpy as np
import matplotlib as mpl



def plot_states_evolution_with_Fneq(b_sphere, r_t_original, r_t_rotated, r_steady, DFneq_t_original, DFneq_t_rotated, c_original='#009000', c_rotated='#6F00FF', c_steady='#ff6600', mapa = 'rainbow', Fneq_x = 0.75, Fneq_y = 1.125, cmap='rainbow_r'):
    
    cmapr = mpl.cm.get_cmap(cmap)

    F_max = np.max([DFneq_t_original.max(), DFneq_t_rotated.max()])
    F_min = np.max([DFneq_t_original.min(), DFneq_t_rotated.min()])

    Nt = len(DFneq_t_original)

    norm_original = mpl.colors.Normalize(vmin=F_min, vmax=F_max)
    m_original = mpl.cm.ScalarMappable(norm=norm_original, cmap=mapa)
    colors_original = [m_original.to_rgba(P) for P in DFneq_t_original]

    norm_rotated = mpl.colors.Normalize(vmin=F_min, vmax=F_max)
    m_rotated = mpl.cm.ScalarMappable(norm=norm_rotated, cmap=mapa)
    colors_rotated = [m_rotated.to_rgba(P) for P in DFneq_t_rotated]    

    pts  = []
    pts.append(r_t_original[:,0])
    pts.append(r_t_rotated[:,0])
    pts.append(r_steady)
    pts = [r_t_original[:,0], r_t_rotated[:,0], r_steady]

    b_sphere.point_color = [mpl.colors.to_rgba(c_original),mpl.colors.to_rgba(c_rotated), mpl.colors.to_rgba(c_steady)]
    b_sphere.point_marker = ['o']
    b_sphere.point_size = [80, 80, 80]
    b_sphere.sphere_alpha = 0.15


    for pt in pts:
        b_sphere.add_points(pt)

    b_sphere.add_vectors(r_t_original[:,0])
    b_sphere.vector_color.append(colors_original[0])
    b_sphere.add_vectors(r_t_rotated[:,0])
    b_sphere.vector_color.append(colors_rotated[0])    

    for i in range(Nt):
        b_sphere.point_color.append(colors_original[i])
        b_sphere.point_size.append(7.5)
        b_sphere.point_marker = ['o']
        b_sphere.add_points(r_t_original[:,i])

    for i in range(Nt):
        b_sphere.point_color.append(colors_rotated[i])
        b_sphere.point_size.append(5)
        b_sphere.point_marker = ['o']
        b_sphere.add_points(r_t_rotated[:,i]) 

    return None