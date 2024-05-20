import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
import pandas as pd

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

def sample_highs(df : pd.DataFrame, labels_column : str, sample_proportion: float, highs_proportion: float, random_state=None):

    # Calculate total rows to sample
    n_highs = int(len(df[df[labels_column] == 1]) * sample_proportion)
    n_lows = int(n_highs * (1 - highs_proportion) / highs_proportion)

    # Perform the sampling
    sampled_df_1s = df[df[labels_column] == 1].sample(n=n_highs, random_state=random_state)
    sampled_df_0s = df[df[labels_column] == 0].sample(n=n_lows, random_state=random_state)

    # Concatenate the sampled DataFrames
    return pd.concat([sampled_df_1s, sampled_df_0s])

def total_proportion(n_highs : np.array, n_lows : np.array, n_points: int, train: np.array = None, high_val = 0.55, low_val = 0.45):
 
    highs = int(n_highs.sum())
    lows = int(n_lows.sum())
    scaling_factor = n_points / (highs + lows)
    highs = int(np.rint(highs * scaling_factor))
    lows = int(np.rint(lows * scaling_factor))

    initial = np.random.permutation(np.repeat([high_val, low_val], [highs, lows]))
    initial = np.where(np.isnan(train), initial, train)

    return train, initial

def like_proportions(areas : np.array, proportions : np.array, train: np.array = None, high_val = 0.55, low_val = 0.45):
    initial = np.zeros_like(areas, dtype=float)
    unique_areas, area_counts = np.unique(areas, return_counts=True)
    for area in unique_areas:
        highs = int(np.rint(proportions[area - 1] * area_counts[area - 1]))
        lows = area_counts[area - 1] - highs
        initial[areas == area] = np.random.permutation(np.repeat([high_val, low_val], [highs, lows]))

    initial = np.where(np.isnan(train), initial, train)

    return train, initial

def like_proportions_train_adjusted(areas: np.array, proportions: np.array, train: np.array = None, high_val=0.55, low_val=0.45):
    unique_areas, area_counts = np.unique(areas, return_counts=True)
    if train is None:
        train = pd.Series(np.nan, index=np.arange(len(areas)))
    initial = train.copy()

    for area in unique_areas:

        total_highs = int(np.rint(proportions[area - 1] * area_counts[area - 1]))
        total_lows = area_counts[area - 1] - total_highs

        area_indices = np.where(areas == area)[0]

        train_highs_indices = np.where(train[area_indices] >= high_val)[0]
        train_lows_indices = np.where(train[area_indices] <= low_val)[0]

        train_highs = len(train_highs_indices)
        train_lows = len(train_lows_indices)

        remaining_highs = total_highs - train_highs
        remaining_lows = total_lows - train_lows

        if remaining_highs < 0:
            # Reduce train highs if they exceed total highs, adjust in train_data
            extra_highs = np.abs(remaining_highs)
            flip_indices = np.random.choice(train_highs_indices, extra_highs, replace=False)
            train[area_indices[flip_indices]] = np.nan
            initial[area_indices[flip_indices]] = np.nan

        if remaining_lows < 0:
            # Similarly, reduce train lows if they exceed total lows
            extra_lows = np.abs(remaining_lows)
            flip_indices = np.random.choice(train_lows_indices, extra_lows, replace=False)
            train[area_indices[flip_indices]] = np.nan
            initial[area_indices[flip_indices]] = np.nan

        new_train_highs_indices = np.where(train[area_indices] == 1)[0]
        new_train_lows_indices = np.where(train[area_indices] == 0)[0]

        new_train_highs = len(new_train_highs_indices)
        new_train_lows = len(new_train_lows_indices)

        new_remaining_highs = total_highs - new_train_highs
        new_remaining_lows = total_lows - new_train_lows

        if new_remaining_highs + new_remaining_lows > 0:
            highs_lows_array = np.random.permutation(np.repeat([high_val, low_val], [new_remaining_highs, new_remaining_lows]))
            initial[area_indices[np.isnan(initial[area_indices])]] = highs_lows_array
        
        assert total_highs == np.sum(initial[area_indices] >= high_val), "Highs do not match"
        assert total_lows == np.sum(initial[area_indices] <= low_val), "Lows do not match"

    return  train, initial

def compute_metrics(labels, predictions, proportion_targets, proportions, classification_metrics, proportion_metrics, unupdatable=None):

    classification_scores, proportion_scores = {}, {}

    if unupdatable is not None:
        valid_indices = ~np.isnan(labels) & ~unupdatable
    else:
        valid_indices = ~np.isnan(labels)


    for metric in classification_metrics:
        value = getattr(metrics, metric)(labels[valid_indices], predictions[valid_indices])
        if type(value) == np.ndarray:
            value = value.tolist()
        classification_scores[metric] = value

    for metric in proportion_metrics:
        proportion_scores[metric] = getattr(metrics, metric)(proportion_targets, proportions)

    return classification_scores, proportion_scores

def get_proportions(vertices, areas):
    num_areas = np.max(areas)
    proportions = np.zeros(shape=num_areas)
    for x in range(1, num_areas + 1):
        proportions[x - 1] = np.nanmean(vertices[areas == x])
    return proportions