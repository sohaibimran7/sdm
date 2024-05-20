from numpy import ndarray, array
from scipy.sparse import lil_matrix
from sdm.neighbours import delaunay, chebyshev, adjust_duplicates

def calc_areas(dim, zoom_factor):
    """
    Calculate the areas (pixels) each sub-pixel belongs to. 

    Args:
        dim (int): The dimension of the grid.
        zoom_factor (int): The zoom factor.

    Returns:
        list: A list of areas (pixels) each sub-pixel belongs to.
    Raises:
        ValueError: If the zoom_factor is not a factor of the grid dimension.

    """
    if not dim % zoom_factor == 0:
        raise ValueError(f"zoom_factor must be a factor of Grid dimension, received zoom_factor: {zoom_factor}")
    ratio = int(dim / zoom_factor)
    areas = []
    for i in range(ratio):
        for z in range(zoom_factor):
            for j in range(1, ratio + 1):
                areas += [ratio * i + j] * zoom_factor

    return array(areas)


class Graph:
    def __init__(self, vertices: ndarray, coordinates, areas: ndarray,
                                unupdatable: ndarray=None, edges=None):
        self.vertices = vertices
        self.coordinates = coordinates
        self.coordinates = adjust_duplicates(coordinates, verbose=False) if coordinates is not None else None
        self.areas = areas
        self.unupdatable = unupdatable
        self.edges = delaunay(lil_matrix((vertices.shape[0], vertices.shape[0])),
                              self.coordinates) if edges is None else edges
        self.num_neighbours = self.edges.sum(axis=0).A.flatten()
        assert (self.num_neighbours == 0).sum() == 0, "There are areas with no neighbours. Please check the graph edges, this is often because of duplicate coordinates, and can be fixed by decreasing the decimal places in the adjust_duplicates function in neighbours.py"
    
    def reset(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Grid(Graph):
    def __init__(self, vertices: ndarray, unupdatable: ndarray=None, zoom_factor=None, edges=None):
        self.dim = vertices.shape[0] ** 0.5
        if not self.dim.is_integer():
            raise ValueError(f"Grid expects a square number of vertices, received {vertices.shape[-1]} vertices")
        else:
            self.dim = int(self.dim)
        self.zoom_factor = self.dim if zoom_factor is None else zoom_factor
        areas = calc_areas(self.dim, self.zoom_factor)
        edges = chebyshev(lil_matrix((vertices.shape[0], vertices.shape[0])), self.dim) if edges is None else edges
        super().__init__(vertices, coordinates=None, areas=areas, unupdatable=unupdatable, edges=edges)
