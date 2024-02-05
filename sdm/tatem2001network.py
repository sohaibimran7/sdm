from graph import Graph, Grid
from neigbours import NeighboursTraverser
from scipy import sparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import csv

np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

class HopfieldNetwork:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.energy = None
        self.proportion_targets = None

    def predict(self, iterations, weights, stopping_threshold, g1_lamda=1, g2_lamda=1, ap_lamda=1,
                proportion_targets=None, plot=True, print_values=False, log_file=None,
                plot_print_and_log_after=1, plots_dir=None):

        # useful for plotting
        if isinstance(self.graph, Grid):
            dim = self.graph.dim
        
        if proportion_targets is not None:
            self.proportion_targets = proportion_targets
        elif self.proportion_targets is None:
            self.set_proportion_targets()

        # plot, print, and log network parameters and targets.
        if plot:
            self.plot(iteration='Target', plots_dir=plots_dir)

        if print_values:
            print("\niterations:", iterations)
            print("\nlamdas:", g1_lamda, g2_lamda, ap_lamda)
            print("\nweights:", weights)
            print("\ntarget vertices:\n", self.graph.vertices.reshape(dim, dim) if isinstance(self.graph, Grid) else self.graph.vertices)
            print("\ntarget area proportions:\n", self.proportion_targets)
            print("\nareas:\n", self.graph.areas.reshape(dim, dim) if isinstance(self.graph, Grid) else self.graph.areas)

            
        if log_file:
            self.log_to_file(log_file,
                             iterations=[iterations],
                             lamdas=[g1_lamda, g2_lamda, ap_lamda],
                             iteration=["target"],
                             target_vertices=self.graph.vertices.reshape(dim, dim) if
                             isinstance(self.graph, Grid) else self.graph.vertices,
                             target_area_proportions=self.proportion_targets,
                             areas=self.graph.areas.reshape(dim, dim) if
                             isinstance(self.graph, Grid) else self.graph.areas,
                             weights=weights,
                             )

        columns, multipliers, counts = self.calc_col_mult_count()

        # random respecting area proportions
        self.graph.vertices = self.graph.vertices.astype('float64')
        areas, area_counts = np.unique(self.graph.areas[self.graph.unupdatable != True], return_counts=True)
        for area in areas:
            highs = np.rint(self.proportion_targets[area - 1] * area_counts[area - 1])
            lows = area_counts[area - 1] - highs
            self.graph.vertices[(self.graph.unupdatable != True) & (self.graph.areas == area)]= np.random.permutation(np.repeat([0.55, 0.45], [highs, lows]))

        # Iterations
        for iteration in range(iterations):

            g1_derivative = self.calc_g1_derivative(columns, multipliers, counts, g1_lamda, log_file)
            g2_derivative = self.calc_g2_derivative(columns, multipliers, counts, g2_lamda, log_file)

            derivatives = self.calc_area_proportions(ap_lamda) - self.proportion_targets
            area_proportion_derivative = np.zeros_like(self.graph.vertices)
            for area in range(1, derivatives.shape[0] + 1):
                area_proportion_derivative[self.graph.areas == area] = derivatives[area-1]

            # energy_derivative = weights.dot(np.vstack((g1_derivative, g2_derivative, area_proportion_derivative)))
            energy_derivative = np.nansum(np.multiply(weights.reshape(-1, 1), np.vstack((g1_derivative, g2_derivative, area_proportion_derivative))), axis = 0)

            # plot, print, and log iterations
            if iteration % plot_print_and_log_after == 0:
                if plot:
                    self.plot(iteration=iteration , plots_dir=plots_dir)

                if print_values:
                    print("\niteration:", iteration)
                    print("\nvertices:\n", self.graph.vertices.reshape(dim, dim) if isinstance(self.graph, Grid) else self.graph.vertices)
                    print("\nmean_vih:\n", self.calc_mean_vih(columns, multipliers, counts).reshape(dim, dim)
                    if isinstance(self.graph, Grid) else self.calc_mean_vih(columns, multipliers, counts))
                    print("\ng1 derivative:\n", g1_derivative.reshape(dim, dim) if isinstance(self.graph, Grid) else g1_derivative)
                    print("\ng2 derivative:\n", g2_derivative.reshape(dim, dim) if isinstance(self.graph, Grid) else g2_derivative)
                    print("\narea proportions:\n", self.calc_area_proportions(ap_lamda))
                    print("\narea proportion derivative:\n", area_proportion_derivative.reshape(dim, dim) if isinstance(self.graph, Grid) else area_proportion_derivative)
                    print("\nenergy derivative:\n", energy_derivative.reshape(dim, dim) if isinstance(self.graph, Grid) else energy_derivative)

                # Logging
                if log_file:
                    self.log_to_file(log_file,
                                     iteration=[iteration],
                                     vertices=self.graph.vertices.reshape(dim, dim) if
                                     isinstance(self.graph, Grid) else self.graph.vertices,
                                     g1_derivative=g1_derivative.reshape(dim, dim),
                                     g2_derivative=g2_derivative.reshape(dim, dim),
                                     area_proportions=self.calc_area_proportions(ap_lamda),
                                     area_proportion_derivative=area_proportion_derivative.reshape(
                                         dim, dim)
                                     if isinstance(self.graph, Grid) else area_proportion_derivative,
                                     energy_derivative=energy_derivative.reshape(
                                         dim, dim)
                                     if isinstance(self.graph, Grid) else energy_derivative,
                                     )

            if np.sum(np.abs(energy_derivative)) < stopping_threshold:
                # print(energy_derivative)
                print(f"converged after {iteration} iterations")
                break

            self.graph.vertices[self.graph.unupdatable != True] -= energy_derivative[self.graph.unupdatable != True]

            # self.graph.vertices = np.where((self.graph.vertices + energy_derivative >= 0) &
            #                                (self.graph.vertices + energy_derivative <= 1),
            #                                self.graph.vertices + energy_derivative,
            #                                self.graph.vertices)

        # plot, print, and log final result
        self.graph.vertices = (self.graph.vertices >= 0.5).astype(int)

        if plot:
            self.plot(iteration='Result'  , plots_dir=plots_dir)

        if print_values:
            print("\niteration:", "result")
            print("\nvertices:\n", self.graph.vertices.reshape(dim, dim) if isinstance(self.graph, Grid) else self.graph.vertices)
            print("\narea_proportions:\n", self.calc_area_proportions(ap_lamda))

        if log_file:
            self.log_to_file(log_file,
                             iteration=["result"],
                             vertices=self.graph.vertices.reshape(dim, dim) if
                             isinstance(self.graph, Grid) else self.graph.vertices,
                             area_proportions = self.calc_area_proportions(ap_lamda)
                             )

    def log_to_file(self, log_file, **kwargs):
        for key, value in kwargs.items():
            pd.DataFrame(['', key]).to_csv(log_file, mode='a', header=False, index=False)
            pd.DataFrame(value).to_csv(log_file, mode='a', header=False, index=False, na_rep='nan')

    def plot(self, iteration, plots_dir=None):
        if isinstance(self.graph, Grid):
            f, ax = plt.subplots()
            lines = [self.graph.zoom_factor * i for i in range(1, int(self.graph.dim/self.graph.zoom_factor) + 1)]
            ax.hlines(lines, xmin=0, xmax=self.graph.dim)
            ax.vlines(lines, ymin=0, ymax=self.graph.dim)
            sns.heatmap(self.graph.vertices.reshape(self.graph.dim, self.graph.dim),
                        cmap='Greys', cbar=True, vmax=1, vmin=0)
            plt.show()
        else:
            if plots_dir is None:
                pd.DataFrame(np.vstack((self.graph.vertices.reshape(1, -1), self.graph.coordinates.T)).T).to_csv(f'iteration: {iteration}.csv')
            else:
                pd.DataFrame(np.vstack((self.graph.vertices.reshape(1, -1), self.graph.coordinates.T)).T).to_csv(f'{plots_dir}/iteration: {iteration}.csv')


    def set_proportion_targets(self):
        num_areas = np.amax(self.graph.areas)
        self.proportion_targets = np.zeros(shape=num_areas)
        for x in range(1, num_areas + 1):
            self.proportion_targets[x - 1] = np.mean(self.graph.vertices[self.graph.areas == x])

    def calc_area_proportions(self, lamda):
        num_areas = np.amax(self.graph.areas)
        area_proportions = np.zeros(shape=num_areas)
        for x in range(1, num_areas + 1):
            area_proportions[x - 1] = 0.5 * np.mean(1 + np.tanh((self.graph.vertices[self.graph.areas == x] - 0.5) * lamda))
        return area_proportions

    def calc_g1_derivative(self, columns, multipliers, counts, lamda, log_file=None):
        mean_vih = self.calc_mean_vih(columns, multipliers, counts)
        # double check
        g1_derivative = 0.5 * (1 + np.tanh((mean_vih - 0.5) * lamda)) * (self.graph.vertices - 1)
        return g1_derivative

    def calc_g2_derivative(self, columns, multipliers, counts, lamda, log_file=None):
        mean_vih = self.calc_mean_vih(columns, multipliers, counts)
        # double check
        g2_derivative = 0.5 * (1 - np.tanh((mean_vih - 0.5) * lamda)) * self.graph.vertices
        return g2_derivative

    def calc_col_mult_count(self):
        rows, columns = np.nonzero(self.graph.edges == 1)
        counts = np.zeros_like(self.graph.vertices)
        unique, c = np.unique(rows, return_counts=True)
        np.put(counts, unique, c)
        j = np.arange(rows.shape[0])
        multipliers = sparse.csr_matrix(((np.ones_like(rows)), (j, rows)),
                                       shape=(rows.shape[0], self.graph.vertices.shape[0]))

        return columns, multipliers, counts

    def calc_mean_vih(self, columns, multipliers, counts):
        mean_vih = sparse.csr_matrix.dot(self.graph.vertices[columns],
                                              multipliers) / counts
        return np.array(mean_vih)

