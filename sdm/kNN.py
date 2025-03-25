#%%
from sdm.graph import Graph
from sklearn.neighbors import KNeighborsClassifier
from sdm.utils import compute_metrics, get_proportions
from dataclasses import dataclass
import numpy as np
import pandas as pd

class kNN:
    def __init__(self, args, graph: Graph, proportion_targets=None, labels=None):
        self.args = args
        self.graph = graph
        self.proportion_targets = proportion_targets
        self.labels = labels
        self.coordinates = graph.coordinates
        self.initial_vertices = graph.vertices.copy()
        self.knn = KNeighborsClassifier(n_neighbors=self.args.k)
        self.fit()

    def fit(self) -> None:
        self.knn.fit(self.coordinates[self.graph.unupdatable], self.labels[self.graph.unupdatable])

    def predict(self) -> np.ndarray:
        predictions = self.knn.predict(self.coordinates)
        # replace predictions with labels where labels is not nan
        predictions[self.graph.unupdatable] = self.labels[self.graph.unupdatable]

        self.graph.vertices = predictions
        classification_scores, proportion_scores = self.evaluation_step()
        return classification_scores, proportion_scores
    
    def evaluation_step(self) -> tuple:
        predictions = (self.graph.vertices >= 0.5).astype(int)

        classification_scores, proportion_scores = compute_metrics(labels=self.labels, 
                                                                    predictions=predictions, 
                                                                    proportion_targets=self.proportion_targets, 
                                                                    proportions=get_proportions(
                                                                        vertices=predictions, 
                                                                        areas=self.graph.areas),
                                                                    classification_metrics=self.args.classification_metrics, 
                                                                    proportion_metrics=self.args.proportion_metrics, 
                                                                    unupdatable=self.graph.unupdatable)

        return classification_scores, proportion_scores


#%%
if __name__ == "__main__":
    @dataclass
    class TestArgs:
        k: int = 5
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
    
    # Initialize and run kNN
    knn_model = kNN(
        args=args,
        graph=graph,
        proportion_targets=proportions_data['ones_proportion'].to_numpy(),
        labels=labels
    )
    
    classification_scores, proportion_scores = knn_model.predict()
    print("\nClassification Scores:")
    print(classification_scores)
    print("\nProportion Scores:")
    print(proportion_scores)
