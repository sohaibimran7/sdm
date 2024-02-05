from numpy import ndarray, array
from scipy.sparse import lil_matrix
from neigbours import delaunay, chebyshev

def calc_areas(dim, zoom_factor):
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
        self.areas = areas
        self.unupdatable = unupdatable
        self.edges = delaunay(lil_matrix((vertices.shape[0], vertices.shape[0])),
                              coordinates) if edges is None else edges


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
