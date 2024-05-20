#%%
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sdm.hnn import HopfieldNetwork
from sdm.graph import Graph, Grid
from sdm.run import HNNArgs
# %%
shared_args = dict(run_name='test', save_results=False)

def test_set_proportion_targets():
    args = HNNArgs(**shared_args)
    graph = Graph(vertices=np.array([0, 1, 0, 1]), coordinates=np.array([[0, 0], [1, 0], [0, 3], [3, 3]]), areas=np.array([1, 1, 2, 2]))
    hnn = HopfieldNetwork(args, graph)
    hnn.set_proportion_targets()
    assert np.array_equal(hnn.proportion_targets, np.array([0.5, 0.5]))

def test_get_neighbourhood_mean():
    args = HNNArgs(**shared_args)
    graph = Grid(vertices=np.array([0, 1, 0, 1]))
    hnn = HopfieldNetwork(args, graph)
    neighbourhood_mean = hnn.get_neighbourhood_mean()
    expected = np.array([2/3, 1/3, 2/3, 1/3])
    assert np.array_equal(neighbourhood_mean, expected)


def test_plot():
    args = HNNArgs(**shared_args)
    graph = Grid(vertices=np.array([0, 1, 0, 1]), zoom_factor=2)
    hnn = HopfieldNetwork(args, graph)
    try:
        hnn.plot(0)
        assert True
    except NotImplementedError:
        assert False
    except Exception as e:
        assert False, f"Unexpected exception raised: {e}"

test_set_proportion_targets()
test_get_neighbourhood_mean()
test_plot()



# %%
