#%%
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sdm.neighbours import chebyshev, adjust_duplicates, delaunay

def test_chebyshev():
    # Test case 1: 3x3 matrix
    edges = lil_matrix((9, 9))
    dim = 3
    result = chebyshev(edges, dim)
    expected_result = np.array([
        [0, 1, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 0]
    ])
    # asset expected result is symmetric
    assert np.array_equal(expected_result.T, expected_result), "Expected result is not symmetric, test is faulty"

    assert np.array_equal(result.toarray().astype(int), expected_result, "function chebyshev is not working as expected")

test_chebyshev()
# %%
def test_adjust_duplicates():
    # Test case 1: No duplicates
    coords1 = np.array([
        [0., 0.],
        [1., 0.],
        [2., 0.],
        [0., 1.],
        [1., 1.],
        [2., 1.],
        [0., 2.],
        [1., 2.],
        [2., 2.],
    ])
    result1 = adjust_duplicates(coords1)
    assert np.array_equal(result1, coords1), "No duplicates, expected result should be the same as input"

    # Test case 2: Duplicates with noise
    coords2 = np.array([
        [1., 0.],
        [2., 0.],
        [0., 1.],
        [2., 1.],
        [0., 2.],
        [1., 2.],
        [2., 2.],
        [0., 0.],  # Duplicate
        [0., 0.],  # Duplicate
        [1., 1.],  # Duplicate
        [1., 1.],  # Duplicate
    ])
    result2 = adjust_duplicates(coords2, decimal_places=4, max_noise=0.1)
    assert np.array_equal(result2[:7], coords2[:7]), "Expected result should be the same for non-duplicate coordinates"
    assert not np.array_equal(result2[7:], coords2[7:]), "Expected result should be different for duplicate coordinates"
test_adjust_duplicates()
# %%
def test_delaunay():
    # Test case 1: 3 points
    edges = lil_matrix((3, 3))
    coordinates = np.array([
        [0., 0.],
        [1., 0.],
        [0., 1.]
    ])
    result = delaunay(edges, coordinates)
    expected_result = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    assert np.array_equal(result.toarray().astype(int), expected_result), "function delaunay is not working as expected"

    # Test case 2: 4 points
    edges = lil_matrix((4, 4))
    coordinates = np.array([
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1., 1.]
    ])
    result = delaunay(edges, coordinates)
    expected_result = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    assert np.array_equal(result.toarray().astype(int), expected_result), "function delaunay is not working as expected"

test_delaunay()
# %%
