#%%
from sdm.graph import Graph, Grid
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sdm.utils import compute_metrics, get_proportions
# import wandb

class HopfieldNetwork:
    def __init__(self, args, graph: Graph, proportion_targets=None, labels=None):

        self.args = args
        self.graph = graph
        self.initial_vertices = self.graph.vertices.copy()
        self.labels = labels
        self.proportion_targets = proportion_targets

        # if self.args.log_to_wandb:
        #     wandb.init(project=self.args.project_name, name=self.args.run_name)
        #     wandb.define_metric("iteration")
        #     wandb.define_metric("*", step_metric="iteration")

    def evaluation_step(self, iteration, energy_derivative=None) -> tuple:
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
        
        # if self.args.log_to_wandb:
        #     wandb.log({"iteration": iteration})
        #     wandb.log({"energy_derivative": energy_derivative})
        #     wandb.log(classification_scores)
        #     wandb.log(proportion_scores)

        return classification_scores, proportion_scores
    
    def prediction_step(self) -> np.float64:
        neighbourhood_mean = self.get_neighbourhood_mean()  
        g1_derivative = self.get_g1_derivative(neighbourhood_mean, self.args.lamda)
        g2_derivative = self.get_g2_derivative(neighbourhood_mean, self.args.lamda)

        derivatives = self.get_proportions_tanh(self.args.lamda) - self.proportion_targets
        area_proportion_derivative = np.zeros_like(self.graph.vertices)
        for area in range(1, derivatives.shape[0] + 1):
            area_proportion_derivative[self.graph.areas == area] = derivatives[area-1]

        energy_derivative = (self.args.alpha * (g1_derivative + g2_derivative) + (1 - self.args.alpha) * area_proportion_derivative) * self.args.dt

        if self.graph.unupdatable is not None:
            self.graph.vertices[~self.graph.unupdatable] -= energy_derivative[~self.graph.unupdatable]
        else:
            self.graph.vertices -= energy_derivative

        return np.mean(np.abs(energy_derivative))

    def predict(self) -> tuple:
        for iteration in range(self.args.n_iter):
            # if self.args.log_to_wandb and (iteration % self.args.log_every_n_iters == 0):
            #     if iteration == 0:
            #         self.evaluation_step(iteration=iteration)
            #     else:
            #         self.evaluation_step(iteration=iteration, energy_derivative=energy_derivative)

            energy_derivative = self.prediction_step()
            if energy_derivative < self.args.stopping_threshold:
                break

        self.graph.vertices = (self.graph.vertices >= 0.5).astype(int)
        classification_scores, proportion_scores = self.evaluation_step(iteration=iteration, energy_derivative=energy_derivative)
        # if self.args.log_to_wandb:  wandb.finish()
        return classification_scores, proportion_scores, energy_derivative, iteration + 1

    def plot(self, iteration) -> None:
        if isinstance(self.graph, Grid):
            f, ax = plt.subplots()
            lines = [self.graph.zoom_factor * i for i in range(1, int(self.graph.dim/self.graph.zoom_factor) + 1)]
            ax.hlines(lines, xmin=0, xmax=self.graph.dim)
            ax.vlines(lines, ymin=0, ymax=self.graph.dim)
            sns.heatmap(self.graph.vertices.reshape(self.graph.dim, self.graph.dim),
                        cmap='Greys', cbar=True, vmax=1, vmin=0)
            plt.show()
        else:
            raise NotImplementedError("Plotting for graphs is not implemented. Please use GIS software to plot the results")

    def get_proportions_tanh(self, lamda) -> np.ndarray:
        num_areas = np.amax(self.graph.areas)
        area_proportions = np.zeros(shape=num_areas)
        for x in range(1, num_areas + 1):
            area_proportions[x - 1] = 0.5 * np.mean(1 + np.tanh((self.graph.vertices[self.graph.areas == x] - 0.5) * lamda))
        return area_proportions

    def get_g1_derivative(self, neighbourhood_mean, lamda) -> np.ndarray:
        return 0.5 * (1 + np.tanh((neighbourhood_mean - 0.5) * lamda)) * (self.graph.vertices - 1)

    def get_g2_derivative(self, neighbourhood_mean, lamda) -> np.ndarray:
        return 0.5 * (1 - np.tanh((neighbourhood_mean - 0.5) * lamda)) * self.graph.vertices

    def get_neighbourhood_mean(self) -> np.ndarray:
        neighborhood_sums = self.graph.edges.dot(self.graph.vertices)
        return neighborhood_sums / self.graph.num_neighbours
