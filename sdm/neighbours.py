import numpy as np
from scipy.spatial import Delaunay


def chebyshev(edges, dim: float):
    """
    Given a (empty) adjacency matrix, and a dim, stores 1 in all elements of the Cth row of the matrix that
    would be neighbouring C if the row was reshaped to a square grid.

    Parameters
    ----------
    edges:          array_like of shape (dim**2, dim**2)
    dim         :   int containing the grid's dimensions (also equivalent to sqrt(edges.shape(0)))

    Returns
    -------
    array_like of shape (dim**2, dim**2)
    """

    for centre in range(edges.shape[0]):

        centre_row = centre // dim
        centre_col = centre % dim

        # | |x| |
        # | |C| |   flatten
        # | | | |   ---->   | |x| | |C| | | | |
        if centre_row - 1 >= 0:
            edges[centre, centre - dim] = 1

            # |x| | |
            # | |C| |   flatten
            # | | | |   ---->   |x| | | |C| | | | |
            if centre_col-1 >= 0:
                edges[centre, centre - dim - 1] = 1

            # | | |x|
            # | |C| |   flatten
            # | | | |   ---->   | | |x| |C| | | | |
            if centre_col + 1 < dim:
                edges[centre, centre - dim + 1] = 1

        # | | | |
        # | |C| |   flatten
        # | |x| |   ---->   | | | | |C| | |x| |
        if centre_row + 1 < dim:
            edges[centre, centre + dim] = 1

            # | | | |
            # | |C| |   flatten
            # |x| | |   ---->   | | | | |C| |x| | |
            if centre_col - 1 >= 0:
                edges[centre, centre + dim - 1] = 1

            # | | | |
            # | |C| |   flatten
            # | | |x|   ---->   | | | | |C| | | |x|
            if centre_col + 1 < dim:
                edges[centre, centre + dim + 1] = 1

        # | | | |
        # |x|C| |   flatten
        # | | | |   ---->   | | | |x|C| | | | |
        if centre_col - 1 >= 0:
            edges[centre, centre - 1] = 1

        # | | | |
        # | |C|x|   flatten
        # | | | |   ---->   | | | | |C|x| | | |
        if centre_col + 1 < dim:
            edges[centre, centre + 1] = 1

    return edges


def delaunay(edges, coordinates):
    """
    Given a (empty) adjacency matrix and a matrix of co-ordinates, finds the delaunay neighbours
    of of all coordinates and stores them in the adjaceny matrix.

    Parameters
    ----------
    edges   : array_like of shape (N, N)
    coordinates: array_like of shape (N, 2)

    Returns
    -------
    array_like of shape (N, N)
    """
    tri = Delaunay(coordinates)
    for simplex in tri.simplices:
        edges[simplex[0], simplex[1]] = 1
        edges[simplex[0], simplex[2]] = 1
        edges[simplex[1], simplex[0]] = 1
        edges[simplex[1], simplex[2]] = 1
        edges[simplex[2], simplex[0]] = 1
        edges[simplex[2], simplex[1]] = 1

    return edges


class NeighboursTraverser:
    def __init__(self):
        self.nearest_neighbours = np.array([])

    def traverse_neighbours(self, edges, lag):
        # Check to see if there are any non-zero values other than principle axis.
        # nnz + edges.shape[0] == edges.shape[0]*edges.shape[1]?

        if self.nearest_neighbours.size == 0:
            self.nearest_neighbours = edges

        new = self.nearest_neighbours.__pow__(lag)
        new.setdiag(0)
        new[new != 0] = lag
        mask = (edges != 0)
        new[mask] = edges[mask]
        return new
