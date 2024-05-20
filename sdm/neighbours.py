import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix

def chebyshev(edges: lil_matrix, dim: float) -> csr_matrix:
    """
    Given an empty adjacency matrix and the dimension of the matrix, returns the adjacency matrix representing the chebyshev neighbours of each cell.

    Parameters:
    edges (lil_matrix): An empty adjacency matrix.
    dim (float): The dimension of the matrix.

    Returns:
    lil_matrix: The adjacency matrix representing the chebyshev neighbours of each cell.
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

    return edges.tocsr()


def delaunay(edges : lil_matrix, coordinates : np.ndarray) -> csr_matrix:
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

    return edges.tocsr()
    
def adjust_duplicates(coords, decimal_places=4, max_noise=1e-3, verbose=True):
    new_coords = coords.copy()
    i = 0
    while True:
        rounded_coords = np.round(new_coords, decimal_places)
        _, indices, counts = np.unique(rounded_coords, axis=0, return_inverse=True, return_counts=True)
        duplicate_mask = counts[indices] > 1

        if verbose:
            print(f"Iteration {i}: {np.sum(duplicate_mask)} duplicates found")

        if not np.any(duplicate_mask):
            break

        # Apply noise only to duplicates
        noise = np.random.uniform(-max_noise, max_noise, new_coords.shape)
        new_coords[duplicate_mask] += noise[duplicate_mask]
        i += 1

    return new_coords
