from matplotlib import cm
import matplotlib as mpl

import qutip


def create_bloch_sphere():
    b = qutip.Bloch(view=[-114,28])
    b.vector_color = ['k','k','k']
    b.add_vectors([1,0,0])
    b.add_vectors([0,1,0])
    b.add_vectors([0,0,1])

    return b

def plot_bloch_vector_in_bloch_sphere(b_sphere, r_vector, q_color):
    b_sphere.point_color = [q_color]
    b_sphere.point_size.append(80)
    b_sphere.point_marker = ['o']
    b_sphere.add_points(r_vector)
    b_sphere.add_vectors(r_vector)
    b_sphere.vector_color.append(q_color)
    return None