from firedrake import *
from numpy.linalg import norm

def locate_dof(x, point):
    """locates dof index from point specified"""

    # obtain cell index, and from it corresponding dofs
    cell = x.function_space().mesh().locate_cell(point)
    cell_dofs = x.function_space().cell_node_list[cell]

    # locate dof closest to point
    coordinates = x.function_space().mesh().coordinates.dat.data
    coordinates_cell_dofs = coordinates[cell_dofs]
    distances = [norm(point-dof_point) for dof_point in coordinates_cell_dofs]
    cell_dof_index = distances.index(min(distances))
    dof = cell_dofs[cell_dof_index]

    return dof


def disturb_dof(x, point, h=1e-3):
    """disturb dof value at point"""

    x.dat.data[locate_dof(x, point)] += h

# Ricker wavelet
def ricker(f, t):
    """ Amplitude for Ricker wavelet """

    mod = 1 - 2*(np.pi*f*(t-1/f))**2
    gauss = np.exp(-(np.pi*f*(t-1/f))**2)

    return mod * gauss
