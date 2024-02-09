#%%
from sdm.graph import Graph, Grid
from scipy import sparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sdm.utils import compute_metrics, get_proportions
import wandb


np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

class HopfieldNetwork:
    def __init__(self, args, graph: Graph, proportion_targets = None, labels = None):
        self.args = args
        self.graph = graph
        self.initial_vertices = self.graph.vertices
        self.labels = labels

        if proportion_targets is not None:
            self.proportion_targets = proportion_targets
        elif self.proportion_targets is None:
            self.set_proportion_targets()

        if self.args.log_to_wandb:
            wandb.init(project=self.args.project_name, name=self.args.run_name)

    def evaluation_step(self, iteration):
            predictions = (self.graph.vertices >= 0.5).astype(int)

            (classification_scores, proportion_scores) = compute_metrics(labels=self.labels, 
                                                                         predictions=predictions, 
                                                                         proportion_targets=self.proportion_targets, 
                                                                         proportions = get_proportions(
                                                                            predictions=predictions, 
                                                                            areas=self.graph.areas),
                                                                         classification_metrics=self.args.classification_metrics, 
                                                                         proportion_metrics=self.args.proportion_metrics, 
                                                                         unupdatable=self.graph.unupdatable)
            
            if self.args.log_to_wandb:

                wandb.log(classification_scores, step=iteration)
                wandb.log(proportion_scores, step=iteration)

            return classification_scores, proportion_scores
    

    def prediction_step(self, columns, multipliers, counts):
        g1_derivative = self.get_g1_derivative(columns, multipliers, counts, self.args.lamdas[0])
        g2_derivative = self.get_g2_derivative(columns, multipliers, counts, self.args.lamdas[1])

        derivatives = self.get_proportions_tanh(self.args.lamdas[2]) - self.proportion_targets
        area_proportion_derivative = np.zeros_like(self.graph.vertices)
        for area in range(1, derivatives.shape[0] + 1):
            area_proportion_derivative[self.graph.areas == area] = derivatives[area-1]

        # energy_derivative = weights.dot(np.vstack((g1_derivative, g2_derivative, area_proportion_derivative)))
        energy_derivative = np.nansum(np.multiply(self.args.weights.reshape(-1, 1), np.vstack((g1_derivative, g2_derivative, area_proportion_derivative))), axis = 0)

        if self.graph.unupdatable is not None:
            self.graph.vertices[~self.graph.unupdatable] -= energy_derivative[~self.graph.unupdatable]
        else:
            self.graph.vertices -= energy_derivative

        return np.sum(np.abs(energy_derivative))

    def predict(self):
            
        columns, multipliers, counts = self.get_col_mult_count()

        # Iterations
        for iteration in range(self.args.n_iter):

            if self.args.log_to_wandb and (iteration % self.args.log_after == 0):
                self.evaluation_step(iteration=iteration)

            delta_energy = self.prediction_step(columns, multipliers, counts)

            if delta_energy < self.args.stopping_threshold:
                # print(energy_derivative)
                print(f"converged after {iteration} iterations")
                break

        self.graph.vertices = (self.graph.vertices >= 0.5).astype(int)

        (classification_scores, proportion_scores) = self.evaluation_step(iteration=iteration)

        if self.args.log_to_wandb:  wandb.finish()
        
        return (classification_scores, proportion_scores)

    # def plot(self, iteration, plots_dir=None):
    #     if isinstance(self.graph, Grid):
    #         f, ax = plt.subplots()
    #         lines = [self.graph.zoom_factor * i for i in range(1, int(self.graph.dim/self.graph.zoom_factor) + 1)]
    #         ax.hlines(lines, xmin=0, xmax=self.graph.dim)
    #         ax.vlines(lines, ymin=0, ymax=self.graph.dim)
    #         sns.heatmap(self.graph.vertices.reshape(self.graph.dim, self.graph.dim),
    #                     cmap='Greys', cbar=True, vmax=1, vmin=0)
    #         plt.show()
    #     else:
    #         raise NotImplementedError("Plotting for graphs is not implemeted. \
    #                                   Please use log_to_file instead and use GIS software to plot the results")

    def set_proportion_targets(self):
        predictions = (self.graph.vertices >= 0.5).astype(int)

        self.proportion_targets = get_proportions(
            predictions=predictions,
            areas=self.graph.areas
        )

    def get_proportions_tanh(self, lamda):
        num_areas = np.amax(self.graph.areas)
        area_proportions = np.zeros(shape=num_areas)
        for x in range(1, num_areas + 1):
            area_proportions[x - 1] = 0.5 * np.mean(1 + np.tanh((self.graph.vertices[self.graph.areas == x] - 0.5) * lamda))
        return area_proportions

    def get_g1_derivative(self, columns, multipliers, counts, lamda):
        mean_vih = self.get_mean_vih(columns, multipliers, counts)
        # double check
        g1_derivative = 0.5 * (1 + np.tanh((mean_vih - 0.5) * lamda)) * (self.graph.vertices - 1)
        return g1_derivative

    def get_g2_derivative(self, columns, multipliers, counts, lamda):
        mean_vih = self.get_mean_vih(columns, multipliers, counts)
        # double check
        g2_derivative = 0.5 * (1 - np.tanh((mean_vih - 0.5) * lamda)) * self.graph.vertices
        return g2_derivative

    def get_col_mult_count(self):
        rows, columns = np.nonzero(self.graph.edges == 1)
        counts = np.zeros_like(self.graph.vertices)
        unique, c = np.unique(rows, return_counts=True)
        np.put(counts, unique, c)
        j = np.arange(rows.shape[0])
        multipliers = sparse.csr_matrix(((np.ones_like(rows)), (j, rows)),
                                       shape=(rows.shape[0], self.graph.vertices.shape[0]))

        return columns, multipliers, counts

    def get_mean_vih(self, columns, multipliers, counts):
        mean_vih = sparse.csr_matrix.dot(self.graph.vertices[columns],
                                              multipliers) / counts
        return np.array(mean_vih)


# %%
