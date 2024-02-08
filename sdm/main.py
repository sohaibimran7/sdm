# %%
import pandas as pd
import numpy as np
import os, json
from dataclasses import dataclass, asdict, field
from sdm.graph import Graph
from sdm.tatem2001network import HopfieldNetwork
from sdm.utils import AdaptiveRepeatedKFold, total_proportion, like_proportions, compute_metrics, get_proportions
from typing import Sequence

# %%
POINTS_DATA_PATH = "data/lancaster/processed/points_data.csv"
PROPORTIONS_DATA_PATH = "data/lancaster/processed/proportions_data.csv"

OUTPUT_DIR = "results/lancaster"

points_data = pd.read_csv(POINTS_DATA_PATH)
proportions_data = pd.read_csv(PROPORTIONS_DATA_PATH)

#%%
@dataclass
class RunArgs():
    """
    Class representing the arguments for running a simulation.
    
    Attributes:
        run_name (str): The name of the simulation run.
        input_data_paths (list): List of input data file paths.
        initialisation (str): The initialization method for the simulation.
        n_runs (int): The number of runs for the simulation.
        use_hnn (bool): Flag indicating whether to use the HNN (Hierarchical Neural Network) model.
    """        
    run_name : str
    input_data_paths = [POINTS_DATA_PATH, PROPORTIONS_DATA_PATH]
    initialisation : str = 'like_proportions'
    n_runs : int = 100
    use_hnn : bool = False
    classification_metrics : Sequence[str] = ('accuracy_score', 'balanced_accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'cohen_kappa_score', 'matthews_corrcoef', 'confusion_matrix')
    proportion_metrics : Sequence[str]= ('mean_squared_error', 'median_absolute_error', 'max_error')


@dataclass
class HNNArgs(RunArgs):
    """
    Class representing the arguments for HNN (Hopfield Neural Network) model.
    
    Attributes:
        use_hnn (bool): Flag indicating whether to use HNN model.
        initial_high_val (int): Initial high value used for initialising the HNN model.
        initial_low_val (int): Initial low value used for initialising the HNN model.
        n_iter (int): Number of iterations for HNN model.
        train_prop (float): Proportion of data used for training.
        cross_validation (bool): Flag indicating whether to use cross-validation.
    """
    use_hnn : bool = True
    initial_high_val : int = 0.55
    initial_low_val : int = 0.45
    n_iter : int = 500
    train_prop: float = 0.0
    cross_validation: bool = False
    weights : np.array = np.array([0.05, 0.05, 1])
    lamdas : np.array = np.array([10, 10, 10])
    stopping_threshold : float = 0.0



def run(points_data : pd.DataFrame, proportions_data: pd.DataFrame, args):
    """
    Runs the Hopfield Neural Network (HNN) algorithm on the given points data.

    Args:
        points_data (pd.DataFrame): The input data containing points information.
        proportions_data (pd.DataFrame): The input data containing proportions information.
        args: The arguments for running the HNN algorithm.

    Returns:
        pd.DataFrame: The updated points data with predicted values.

    Raises:
        AssertionError: If the arguments are not of the correct type or value.

    Notes:
        points_data must contain the following columns: point_id, area_num, latitude, longitude, label.
        proportions_data must contain the following columns: area_num, ones_proportion, n_ones, n_zeros.
    """

    def initialise(high_val, low_val):
        """
        Initializes the values based on the specified initialisation method.

        Parameters:
        - high_val (int): The high value.
        - low_val (int): The low value.

        Returns:
        - numpy.ndarray: The initialized values.

        Raises:
        - NotImplementedError: If the specified initialisation method is not implemented.
        """
        if args.initialisation == 'like_proportions':
            return like_proportions(areas = points_data["area_num"].to_numpy(dtype = int),
                                    proportions = proportions_data["ones_proportion"].to_numpy(), 
                                    high_val = high_val, 
                                    low_val = low_val)
        elif args.initialisation == 'total_proportions':
            return total_proportion(n_ones = proportions_data["n_ones"].to_numpy(dtype = int),
                                    n_zeros = proportions_data["n_zeros"].to_numpy(), 
                                    high_val = high_val, 
                                    low_val= low_val)
        else:
            raise NotImplementedError(f"{args.initialisation} is not an implemented initialisation method")      

    def run_hnn(i: int):
        """
        Runs the Hopfield Neural Network (HNN) for a given iteration.

        Args:
            i (int): The iteration number.

        Returns:
            None
        """
        initial = initialise(high_val=args.initial_high_val, low_val=args.initial_low_val)
        initial = np.where(np.isnan(points_data[f'train_{i + 1}']), initial, points_data[f'train_{i + 1}'])

        points_data[f'initial_{i+1}'] = initial

        hnn = HopfieldNetwork(
            args=args,
            graph=Graph(
                vertices=initial,
                unupdatable=points_data[f'train_{i + 1}'].notna().to_numpy(),
                coordinates=points_data[["latitude", "longitude"]].to_numpy(),
                areas=points_data["area_num"].to_numpy(dtype=int),
            ),
            proportion_targets=proportions_data["ones_proportion"].to_numpy(),
            labels=points_data['label'].to_numpy(),
        )

        classification_scores, proportion_scores = hnn.predict()

        # inplace update of points_data
        points_data[f'predicted_{i+1}'] = hnn.graph.vertices

        return classification_scores, proportion_scores



    # Main function body
    if isinstance(args, HNNArgs):
        assert args.use_hnn, "args.use_hnn must be true where args are an instance of HNNArgs"
    else:
        assert not args.use_hnn, "args must be an instance of HNNArg where args.use_hnn is true"

    multirun_classification_and_proportion_scores = []

    if args.use_hnn:

        labels_nonan = points_data[["point_id", "label"]].dropna()

        if args.cross_validation:

            arkfold = AdaptiveRepeatedKFold(args.n_runs, args.train_prop)

            for i, split in enumerate(arkfold.split(labels_nonan["label"].to_numpy())):
                
                points_data = points_data.merge(labels_nonan.iloc[split[0]]
                                .rename(columns={'label':f'train_{i+1}'})
                                    , on='point_id', how='left')
                
                # run HNN updated points_data in place to prevent repeating code
                (classification_scores, proportion_scores) = run_hnn(i=i)
                multirun_classification_and_proportion_scores.append((classification_scores, proportion_scores))

        else:
            for i in range(args.n_runs):

                points_data = points_data.merge(labels_nonan
                                .sample(frac=args.train_prop)
                                .rename(columns={"label":f"train_{i+1}"})
                                , on='point_id', how='left')
                
                # run HNN updated points_data in place to prevent repeating code
                (classification_scores, proportion_scores) = run_hnn(i=i)
                multirun_classification_and_proportion_scores.append((classification_scores, proportion_scores))
    else:
        for i in range(args.n_runs):
            predictions = initialise(high_val=1, low_val=0)
            points_data[f'predicted_{i+1}'] = predictions
            (classification_scores, proportion_scores) = compute_metrics(labels=points_data['label'],
                                                                       predictions=predictions,
                                                                       proportion_targets=proportions_data['ones_proportion'], 
                                                                       proportions=get_proportions(
                                                                           predictions=predictions, 
                                                                           areas=points_data["area_num"].to_numpy(dtype=int)),
                                                                       classification_metrics=args.classification_metrics,
                                                                       proportion_metrics=args.proportion_metrics)
            
            multirun_classification_and_proportion_scores.append((classification_scores, proportion_scores))

    return points_data, multirun_classification_and_proportion_scores

#%%
args = HNNArgs(run_name='example_run', n_runs=1, use_hnn=True, n_iter=500, train_prop=0, cross_validation=False)

results, multirun_classification_and_proportion_scores = run(points_data=points_data, proportions_data=proportions_data, args=args)
results
#%%
output_path = OUTPUT_DIR + args.run_name
os.makedirs(output_path)
with open(f'{output_path}/config.json', 'x') as fp:
    json.dump(asdict(args), fp)
with open(f'{output_path}/metrics.json', 'x') as fp:
    json.dump(multirun_classification_and_proportion_scores, fp)

results.to_csv(f'{output_path}/predictions.csv')