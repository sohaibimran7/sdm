#%%
from sdm.graph import Graph, Grid, calc_areas
from sdm.neighbours import adjust_duplicates
import numpy as np

import pytest

def test_calc_areas():
    # Test case 1: Valid input
    dim = 4
    zoom_factor = 2
    expected_areas = [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
    areas = calc_areas(dim, zoom_factor)
    assert all([a == b for a, b in zip(areas, expected_areas)]), "calc_areas did not return the expected areas for valid input"

    # Test case 2: zoom_factor not a factor of dim
    dim = 5
    zoom_factor = 2
    with pytest.raises(ValueError) as excinfo:
        calc_areas(dim, zoom_factor)
    assert "zoom_factor must be a factor of Grid dimension" in str(excinfo.value), "calc_areas did not raise ValueError when zoom_factor is not a factor of dim"

test_calc_areas()

# %%
