#%%
from sdm.graph import Graph
from sdm.utils import compute_metrics, get_proportions
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tqdm import tqdm

class AreaToPointKriging:
    def __init__(self, args, graph: Graph, proportion_targets=None, labels=None):
        self.args = args
        self.graph = graph
        self.proportion_targets = proportion_targets
        self.labels = labels
        self.coordinates = graph.coordinates
        self.initial_vertices = graph.vertices.copy()
        
        # Initialize Gaussian Process with RBF kernel
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            random_state=42
        )
        self.fit()

    def fit(self) -> None:
        # Get training data
        train_coords = self.coordinates[self.graph.unupdatable]
        train_labels = self.labels[self.graph.unupdatable]
        
        # Normalize coordinates to prevent numerical issues
        self.coord_mean = train_coords.mean(axis=0)
        self.coord_std = train_coords.std(axis=0)
        train_coords_norm = (train_coords - self.coord_mean) / self.coord_std
        
        # Fit the model
        self.gpr.fit(train_coords_norm, train_labels)

    def predict(self) -> np.ndarray:
        # Normalize coordinates for prediction
        coords_norm = (self.coordinates - self.coord_mean) / self.coord_std
        
        # Predict in batches to avoid memory issues
        batch_size = 1000
        predictions = np.zeros(len(self.coordinates))
        
        # Use tqdm to add a progress bar
        for i in tqdm(range(0, len(coords_norm), batch_size), desc="Predicting"):
            batch_coords = coords_norm[i:i + batch_size]
            predictions[i:i + batch_size] = self.gpr.predict(batch_coords)
        
        # Replace predictions with known labels
        predictions[self.graph.unupdatable] = self.labels[self.graph.unupdatable]
        
        # Apply area proportion constraints
        predictions = self.adjust_for_area_proportions(predictions)
        
        self.graph.vertices = predictions
        classification_scores, proportion_scores = self.evaluation_step()
        return classification_scores, proportion_scores
    
    def adjust_for_area_proportions(self, predictions):
        """Adjust predictions to match target area proportions"""
        adjusted_predictions = predictions.copy()
        
        for area in range(1, len(self.proportion_targets) + 1):
            area_mask = self.graph.areas == area
            area_predictions = predictions[area_mask]
            
            if len(area_predictions) == 0:
                continue
                
            # Sort predictions to find threshold
            sorted_preds = np.sort(area_predictions)
            target_ones = int(np.round(len(area_predictions) * self.proportion_targets[area-1]))
            
            if target_ones > 0:
                threshold = sorted_preds[-target_ones]
                adjusted_predictions[area_mask] = (predictions[area_mask] >= threshold).astype(float)
            
        return adjusted_predictions
    
    def evaluation_step(self) -> tuple:
        predictions = (self.graph.vertices >= 0.5).astype(int)

        classification_scores, proportion_scores = compute_metrics(
            labels=self.labels, 
            predictions=predictions, 
            proportion_targets=self.proportion_targets, 
            proportions=get_proportions(
                vertices=predictions, 
                areas=self.graph.areas
            ),
            classification_metrics=self.args.classification_metrics, 
            proportion_metrics=self.args.proportion_metrics, 
            unupdatable=self.graph.unupdatable
        )

        return classification_scores, proportion_scores

#%%
if __name__ == "__main__":
    @dataclass
    class TestArgs:
        train_prop: float = 0.2
        classification_metrics: tuple = ('accuracy_score', 'balanced_accuracy_score', 'f1_score')
        proportion_metrics: tuple = ('mean_absolute_error', 'median_absolute_error')

    # Initialize args first
    args = TestArgs()

    # Load test data
    points_data = pd.read_csv('data/lancaster/processed/points_data.csv')
    proportions_data = pd.read_csv('data/lancaster/processed/proportions_data.csv')
    
    # Get all labels (ground truth)
    labels = points_data['label'].to_numpy()
    
    # Create training mask
    train_mask = np.zeros_like(labels, dtype=bool)
    valid_mask = ~np.isnan(labels)
    train_indices = np.where(valid_mask)[0]
    np.random.shuffle(train_indices)
    n_train = int(len(train_indices) * args.train_prop)
    train_mask[train_indices[:n_train]] = True
    
    # Create test graph with unupdatable mask for training points
    graph = Graph(
        vertices=np.zeros_like(labels),
        coordinates=points_data[["latitude", "longitude"]].to_numpy(),
        areas=points_data["area_num"].to_numpy(dtype=int),
        unupdatable=train_mask
    )
    
    # Initialize and run AreaToPointKriging
    kriging_model = AreaToPointKriging(
        args=args,
        graph=graph,
        proportion_targets=proportions_data['ones_proportion'].to_numpy(),
        labels=labels
    )
    
    classification_scores, proportion_scores = kriging_model.predict()
    print("\nClassification Scores:")
    print(classification_scores)
    print("\nProportion Scores:")
    print(proportion_scores)