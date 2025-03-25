# %%
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from sdm.graph import Graph
from sdm.hnn import HopfieldNetwork
from sdm.utils import AdaptiveRepeatedKFold, total_proportion, like_proportions, like_proportions_train_adjusted, compute_metrics, get_proportions
from typing import Sequence
from tqdm import tqdm
import warnings
import geopandas as gpd
from sdm.kNN import kNN
import os
import json

#%%
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# #%%
@dataclass
class RunArgs():
    run_name : str
    project_name : str = 'sdm'
    points_data_path : str = 'data/lancaster/processed/points_data.csv'
    use_seperate_proportions : bool = True
    proportions_data_path : str = 'data/lancaster/processed/proportions_data.csv'
    save_results : bool = True
    output_dir : str = 'results/lancaster'
    initialisation : str = 'like_proportions_train_adjusted'
    n_runs : int = 100
    classification_metrics : Sequence[str] = ('accuracy_score', 'balanced_accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'cohen_kappa_score', 'matthews_corrcoef', 'confusion_matrix')
    proportion_metrics : Sequence[str]= ('mean_absolute_error', 'median_absolute_error', 'max_error')
    filter_to_aoi : bool = False
    # Flood map zone 3 can be downloaded from https://www.data.gov.uk/dataset/bed63fc1-dd26-4685-b143-2941088923b3/flood-map-for-planning-rivers-and-sea-flood-zone-3
    aoi_path : str = 'data/lancaster/floodmapR&S_zone3_lancaster/floodmap.shp'
    points_data_epsg : int = 4326


@dataclass
class HNNArgs(RunArgs):
    initial_high_val : int = 0.55
    initial_low_val : int = 0.45
    n_iter : int = 200
    train_prop: float = 0.0
    cross_validation: bool = False
    dt: float = 1.0
    alpha : float = 0.5
    lamda : float = 100.0
    stopping_threshold : float = 0.0
    log_to_wandb : bool = False
    log_every_n_iters : int = 10

@dataclass
class KNNArgs(RunArgs):
    k: int = 5
    train_prop: float = 0.2
    cross_validation: bool = False

def run(args):
    def initialise(high_val, low_val):
        if args.initialisation == 'like_proportions_train_adjusted':
            return like_proportions_train_adjusted(areas = points_data["area_num"].to_numpy(dtype = int),
                                    proportions = proportions,
                                    train = points_data[f'sampled_to_train_{i + 1}'].to_numpy(),
                                    high_val = high_val, 
                                    low_val = low_val)
        elif args.initialisation == 'like_proportions':
            return like_proportions(areas = points_data["area_num"].to_numpy(dtype = int),
                                    proportions = proportions,
                                    train = points_data[f'sampled_to_train_{i + 1}'].to_numpy(),
                                    high_val = high_val, 
                                    low_val = low_val)
        elif args.initialisation == 'total_proportions':
            return total_proportion(n_ones = n_ones,
                                    n_zeros = n_zeros,
                                    train = points_data[f'sampled_to_train_{i + 1}'].to_numpy(),
                                    n_points = len(points_data),
                                    high_val = high_val, 
                                    low_val= low_val)
        else:
            raise NotImplementedError(f"{args.initialisation} is not an implemented initialisation method")      

    def run_hnn(graph, i: int):
        train, initial = initialise(high_val=args.initial_high_val, low_val=args.initial_low_val)

        points_data[f'train_{i+1}'] = train
        points_data[f'initial_{i+1}'] = initial

        # inplace reset of graph attributes instead of creating a new graph object to avoid computing delaunay triangulation again
        graph.reset(vertices=initial, unupdatable=~np.isnan(train))

        # create new HNN object for each iteration
        hnn = HopfieldNetwork(
            args=args,
            graph=graph,
            proportion_targets=proportions,
            labels=points_data['label'].to_numpy(),
        )

        classification_scores, proportion_scores, energy_derivative, n_iter = hnn.predict()
        multirun_classification_and_proportion_scores.append((classification_scores, proportion_scores, energy_derivative, n_iter))

        # inplace update of points_data
        points_data[f'predicted_{i+1}'] = hnn.graph.vertices

    def run_knn(graph, i: int):
        train = points_data[f'sampled_to_train_{i+1}'].to_numpy()
        points_data[f'train_{i+1}'] = train

        # inplace reset of graph attributes
        graph.reset(vertices=np.zeros_like(train), unupdatable=~np.isnan(train))

        # create new kNN object for each iteration
        knn_model = kNN(
            args=args,
            graph=graph,
            proportion_targets=proportions,
            labels=points_data['label'].to_numpy(),
        )

        classification_scores, proportion_scores = knn_model.predict()
        multirun_classification_and_proportion_scores.append((classification_scores, proportion_scores))

        # inplace update of points_data
        points_data[f'predicted_{i+1}'] = knn_model.graph.vertices

    # Main function body
    

    # Prepare output directory
    if args.save_results:
        output_path = args.output_dir + '/' + args.run_name
        os.makedirs(output_path)

    # read data
    points_data = pd.read_csv(args.points_data_path)
    if args.use_seperate_proportions:
        assert args.proportions_data_path is not None, "proportions_data_path is required when use_seperate_proportions is True"
        proportions_data = pd.read_csv(args.proportions_data_path)
        proportions = proportions_data['ones_proportion'].to_numpy()
        n_ones = proportions_data["n_ones"].to_numpy(dtype = int)
        n_zeros = proportions_data["n_zeros"].to_numpy(dtype = int)
    else:
        assert args.initialisation != 'total_proportions', "total_proportions initialisation method is not supported when use_seperate_proportions is False"
        proportions = get_proportions(vertices=points_data['label'].to_numpy(), areas=points_data["area_num"].to_numpy(dtype=int))

    # if args.hnn is True, multirun_classification_and_proportion_scores will be populated inside the run_hnn closure
    multirun_classification_and_proportion_scores = []
    
    if isinstance(args, KNNArgs):
        # create graph
        graph = Graph(
            vertices=np.zeros_like(points_data['label'].to_numpy()),
            coordinates=points_data[["latitude", "longitude"]].to_numpy(),
            areas=points_data["area_num"].to_numpy(dtype=int),
        )

        labels_df_nonan = points_data[["point_id", "label"]].dropna()

        if args.cross_validation:
            arkfold = AdaptiveRepeatedKFold(args.n_runs, args.train_prop)
            
            pbar = tqdm(total=args.n_runs, desc=args.run_name, leave=False)

            for i, split in enumerate(arkfold.split(labels_df_nonan["label"].to_numpy())):
                points_data = points_data.merge(labels_df_nonan.iloc[split[0]]
                                .rename(columns={'label':f'sampled_to_train_{i+1}'})
                                    , on='point_id', how='left')
                
                run_knn(graph=graph, i=i)
                pbar.update(1)
            pbar.close()

        else:
            for i in tqdm(range(args.n_runs), desc=args.run_name, leave=False):
                points_data = points_data.merge(labels_df_nonan
                                .sample(frac=args.train_prop)
                                .rename(columns={"label":f"sampled_to_train_{i+1}"})
                                , on='point_id', how='left')
                
                run_knn(graph=graph, i=i)
            
    elif isinstance(args, HNNArgs):

        # create graph
        graph=Graph(
                vertices=np.zeros_like(points_data['label'].to_numpy()),
                coordinates=points_data[["latitude", "longitude"]].to_numpy(),
                areas=points_data["area_num"].to_numpy(dtype=int),
            )

        labels_df_nonan = points_data[["point_id", "label"]].dropna()

        if args.cross_validation:

            arkfold = AdaptiveRepeatedKFold(args.n_runs, args.train_prop)

            pbar = tqdm(total=args.n_runs, desc=args.run_name, leave=False)

            for i, split in enumerate(arkfold.split(labels_df_nonan["label"].to_numpy())):
    
                points_data = points_data.merge(labels_df_nonan.iloc[split[0]]
                                .rename(columns={'label':f'sampled_to_train_{i+1}'})
                                    , on='point_id', how='left')
                
                # run HNN updates points_data in place to prevent repeating code
                run_hnn(graph=graph, i=i)

                pbar.update(1)
            pbar.close()

        else:
            for i in tqdm(range(args.n_runs), desc=args.run_name, leave=False):

                points_data = points_data.merge(labels_df_nonan
                                .sample(frac=args.train_prop)
                                .rename(columns={"label":f"sampled_to_train_{i+1}"})
                                , on='point_id', how='left')
                
                # run HNN updates points_data in place to prevent repeating code
                run_hnn(graph=graph, i=i)
    else:
        for i in tqdm(range(args.n_runs), desc=args.run_name):
            predictions, _ = initialise(high_val=1, low_val=0)
            points_data[f'predicted_{i+1}'] = predictions
            classification_scores, proportion_scores = compute_metrics(labels=points_data['label'],
                                                                       predictions=predictions,
                                                                       proportion_targets=proportions, 
                                                                       proportions=get_proportions(
                                                                           vertices=predictions, 
                                                                           areas=points_data["area_num"].to_numpy(dtype=int)),
                                                                       classification_metrics=args.classification_metrics,
                                                                       proportion_metrics=args.proportion_metrics)
            
            multirun_classification_and_proportion_scores.append((classification_scores, proportion_scores))

    if args.filter_to_aoi:
        assert args.aoi_path is not None, "aoi_path is required when filter_to_aoi is True"

        aoi = gpd.read_file(args.aoi_path).to_crs(epsg=4326)
        points_data.crs = aoi.crs
        
        # Get the original column names from points_data before the spatial join
        original_columns = points_data.columns.tolist()

        points_data = gpd.GeoDataFrame(points_data, geometry=gpd.points_from_xy(points_data.longitude, points_data.latitude))
        points_data = gpd.sjoin(points_data, aoi, how="inner", op='intersects')

        # Remove the geometry column and any columns added by the aoi GeoDataFrame
        columns_to_drop = [col for col in points_data.columns if col not in original_columns]
        points_data = points_data.drop(columns=columns_to_drop)

        # Convert the GeoDataFrame to a regular pandas DataFrame
        points_data = pd.DataFrame(points_data)

    # save results
    if args.save_results:
        points_data.to_csv(f'{output_path}/predictions.csv')
        with open(f'{output_path}/config.json', 'x') as fp:
            json.dump(asdict(args), fp)
        with open(f'{output_path}/metrics.json', 'x') as fp:
            json.dump(multirun_classification_and_proportion_scores, fp)
    
    return points_data, multirun_classification_and_proportion_scores

#%%
if __name__ == "__main__":

    # Load the predictions CSV
    predictions_hnn = pd.read_csv('results/lancaster/hnn_alpha-0.1_train-0.2/predictions.csv')
    predictions_knn = pd.read_csv('results/lancaster/k-1_train-0.2/predictions.csv')

    def calculate_area_stats(df, method_name, run):
        # Initialize a list to store the results
        area_results = []
        
        # Get unique area IDs
        unique_areas = df['area_num'].unique()

        # Iterate over each area
        for area in tqdm(unique_areas, desc=f"Calculating stats for {method_name}"):
            area_data = df[df['area_num'] == area]
            
            # Calculate actual proportions
            total_points = len(area_data)
            actual_prop = (area_data['label'] == 1).sum() / total_points * 100
            
            # Calculate average predicted proportions across all runs
            pred_props = []
            seed_props = []

            predictions = area_data[f'predicted_{run}']
            pred_prop = (predictions >= 0.5).sum() / total_points * 100
            pred_props.append(pred_prop)
            
            # Get seed/train proportions for this run
            train_zeros = (area_data[f'sampled_to_train_{run}'] == 0).sum() 
            train_ones = (area_data[f'sampled_to_train_{run}'] == 1).sum()
            train_total = train_zeros + train_ones
            if train_total > 0:
                seed_prop = train_ones / train_total * 100
                seed_props.append(seed_prop)

            
            # Average the proportions
            avg_pred_prop = np.mean(pred_props)
            avg_seed_prop = np.mean(seed_props) if seed_props else np.nan
            
            # Append the results
            area_results.append({
                'Area': area,
                'Actual %': round(actual_prop, 2),
                f'{method_name} Predicted %': round(avg_pred_prop, 2),
                f'{method_name} Seed %': round(avg_seed_prop, 2)
            })

        return pd.DataFrame(area_results)

    # Calculate stats for both methods
    hnn_stats = calculate_area_stats(predictions_hnn, 'HNN', run=1)
    knn_stats = calculate_area_stats(predictions_knn, 'kNN', run=1)

    # Merge the results on the 'Area' column
    results = pd.merge(hnn_stats, knn_stats, on='Area', suffixes=('_HNN', '_kNN'))

    # Display the results
    print("\nComparison of Non-residential Percentages by Area:")
    print(results)
# %%
