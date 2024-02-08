import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics

class AdaptiveRepeatedKFold():
    """
    A wrapper class for sklearn.model_selection.RepeatedKFold that allows creating folds based on train_prop and
     n_runs, instead of n_splits and n_repeats.

    Parameters:
    - n_runs : int
        The total number of runs.
    - train_prop : float
        The proportion of data to be used for training. Should be between 0 and 1.
    - dec_places : int, optional
        The number of decimal places to round the train_prop when calculating the test proportion.
        Default is 10.
    - random_state : int or RandomState, optional
        The seed value used by the random number generator. Default is None.

    Methods:
    - get_metadata_routing()
        Not implemented.
    - get_n_splits(X=None, y=None, groups=None)
        refer to sklearn.model_slection.RepeatedKFold.get_n_splits for documentation.
    - split(X, y=None, groups=None)
        refer to sklearn.model_slection.RepeatedKFold.get_n_splits for documentation.

    Raises:
    - AssertionError
        If train_prop or (1 - train_prop) is not a factor of 1, or if train_prop and 1 - train_prop
        are less than 1/n_runs, or if 1/train_prop is not a factor of n_runs.
    - NotImplementedError
        If the get_metadata_routing method is called.

    Notes:
    - The methods are based on the RepeatedKFold class from the scikit-learn library.
    """

    def __init__(self, n_runs: int, train_prop: float, dec_places: int = 10, random_state=None) -> None:
        self.reverse = True
        if train_prop > 0.5:
            self.reverse = False
            train_prop = round(1 - train_prop, dec_places)

        assert (1 / train_prop).is_integer(), "either train_prop or (1 - train_prop) must be a factor of 1"
        self.n_splits = int(1 / train_prop)

        assert self.n_splits <= n_runs, "train_prop and 1 - train_prop can not be less than 1/n_runs"

        assert n_runs % self.n_splits == 0, "1/train_prop must be a factor of n_runs"
        self.n_repeats = int(n_runs / self.n_splits)

        # print(f"{self.n_splits=}, {self.n_repeats=}, {self.reverse=}") # for debugging
        self.kfold = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=random_state)

    def get_metadata_routing(self):
        raise NotImplementedError()

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.kfold.get_n_splits(X, y, groups)

    def split(self, X, y=None, groups=None):
        for train_index, test_index in self.kfold.split(X, y, groups):
            yield (test_index, train_index) if self.reverse else (train_index, test_index)


def total_proportion(n_ones : np.array, n_zeros : np.array, high_val = 0.55, low_val = 0.45):
    """
    Calculates the total proportion of high and low values based on the number of ones and zeros.

    Parameters:
    n_ones (np.array): Array containing the number of ones.
    n_zeros (np.array): Array containing the number of zeros.
    high_val (float, optional): Value assigned to high proportion. Defaults to 0.55.
    low_val (float, optional): Value assigned to low proportion. Defaults to 0.45.

    Returns:
    np.array: Array of randomly permuted high and low values based on the proportions.
    """
    highs = int(n_ones.sum())
    lows = int(n_zeros.sum())
    return np.random.permutation(np.repeat([high_val, low_val], [highs, lows]))

def like_proportions(areas : np.array, proportions : np.array, high_val = 0.55, low_val = 0.45):
    """
    Generate an initial array of values based on the proportions of areas.

    Parameters:
    areas (np.array): An array of areas.
    proportions (np.array): An array of proportions corresponding to each area.
    high_val (float): The value to assign to high proportions (default is 0.55).
    low_val (float): The value to assign to low proportions (default is 0.45).

    Returns:
    np.array: An array of initial values based on the proportions of areas.
    """
    unique_areas, area_counts = np.unique(areas, return_counts=True)
    initial = np.zeros_like(areas, dtype=float)
    for area in unique_areas:
        highs = int(np.rint(proportions[area - 1] * area_counts[area - 1]))
        lows = area_counts[area - 1] - highs
        initial[areas == area] = np.random.permutation(np.repeat([high_val, low_val], [highs, lows]))

    return initial

def compute_metrics(labels, predictions, proportion_targets, proportions, classification_metrics, proportion_metrics, unupdatable=None):
    """
    Perform evaluation step for the network.

    Args:
        classification_metrics (List[str]): List of classification metrics to calculate.
        proportion_metrics (List[str]): List of proportion metrics to calculate.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Tuple containing classification scores and proportion scores.

    Raises:
        AttributeError: If the specified metric is not an attribute of sklearn.metrics.
    
    Notes:
    The metrics are computed by calling sklearn.metrics.<metric>.  
    This function will give incorrect results or throw an error for metrics that require parameters other than y_true and y_pred in sklearn.metrics.
    This function will throw an error if the specified metric is not a 'function in sklearn.metrics.
    """
    classification_scores, proportion_scores = {}, {}

    if unupdatable is not None:
        valid_indices = ~np.isnan(labels) & ~unupdatable
    else:
        valid_indices = ~np.isnan(labels)


    for metric in classification_metrics:
        classification_scores[metric] = getattr(metrics, metric)(labels[valid_indices], predictions[valid_indices])

    for metric in proportion_metrics:
        proportion_scores[metric] = getattr(metrics, metric)(proportion_targets, proportions)

    return (classification_scores, proportion_scores)

def get_proportions(predictions, areas):
    num_areas = np.max(areas)
    proportions = np.zeros(shape=num_areas)
    for x in range(1, num_areas + 1):
        proportions[x - 1] = np.mean(predictions[areas == x])
    return proportions