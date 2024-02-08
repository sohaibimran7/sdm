#%%
from sdm.graph import Graph, Grid
from scipy import sparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sdm.utils import compute_metrics, get_proportions


np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

class HopfieldNetwork:
    def __init__(self, args, graph: Graph, proportion_targets = None, labels = None):
        self.args = args
        self.graph = graph
        self.initial = self.graph.vertices
        self.labels = labels

        if proportion_targets is not None:
            self.proportion_targets = proportion_targets
        elif self.proportion_targets is None:
            self.set_proportion_targets()

        self.energy = None

        # if self.args.use_wandb:
        #     wandb.init(project=self.args.wandb_project, name=self.args.run_name)


    def evaluation_step(self):
            predictions = (self.graph.vertices >= 0.5).astype(int)

            return compute_metrics(labels=self.labels, 
                                   predictions=predictions, 
                                   proportion_targets=self.proportion_targets, 
                                   proportions = get_proportions(
                                        predictions=predictions,
                                        areas=self.graph.areas),
                                   classification_metrics=self.args.classification_metrics, 
                                   proportion_metrics=self.args.proportion_metrics, 
                                   unupdatable=self.graph.unupdatable)

    def prediction_step(self, columns, multipliers, counts):
        g1_derivative = self.get_g1_derivative(columns, multipliers, counts, self.args.lamdas[0])
        g2_derivative = self.get_g2_derivative(columns, multipliers, counts, self.args.lamdas[1])

        derivatives = self.get_proportions_tanh(self.args.lamdas[2]) - self.proportion_targets
        area_proportion_derivative = np.zeros_like(self.graph.vertices)
        for area in range(1, derivatives.shape[0] + 1):
            area_proportion_derivative[self.graph.areas == area] = derivatives[area-1]

        # energy_derivative = weights.dot(np.vstack((g1_derivative, g2_derivative, area_proportion_derivative)))
        energy_derivative = np.nansum(np.multiply(self.args.weights.reshape(-1, 1), np.vstack((g1_derivative, g2_derivative, area_proportion_derivative))), axis = 0)

        self.graph.vertices[~self.graph.unupdatable] -= energy_derivative[~self.graph.unupdatable]

        return np.sum(np.abs(energy_derivative))

    def predict(self):
            
        columns, multipliers, counts = self.get_col_mult_count()

        # Iterations
        for iteration in range(self.args.n_iter):

            delta_energy = self.prediction_step(columns, multipliers, counts)

            if delta_energy < self.args.stopping_threshold:
                # print(energy_derivative)
                print(f"converged after {iteration} iterations")
                break

        self.graph.vertices = (self.graph.vertices >= 0.5).astype(int)
        
        return self.evaluation_step()
            
        

        # # useful for plotting
        # if isinstance(self.graph, Grid):
        #     dim = self.graph.dim

        # # plot, print, and log network parameters and targets.
        # if self.args.plot:
        #     self.plot(iteration='Target', plots_dir=self.args.plots_dir)

        # if self.args.print_values:
        #     print("\niterations:", self.args.n_iter)
        #     print("\nlamdas:", self.args.lamdas[0], self.args.lamdas[1], self.args.lamdas[2])
        #     print("\nweights:", self.args.weights)
        #     print("\ntarget vertices:\n", self.graph.vertices.reshape(dim, dim) if isinstance(self.graph, Grid) else self.graph.vertices)
        #     print("\ntarget area proportions:\n", self.proportion_targets)
        #     print("\nareas:\n", self.graph.areas.reshape(dim, dim) if isinstance(self.graph, Grid) else self.graph.areas)

            
        # if self.args.log_file:
        #     self.log_to_file(self.args.log_file,
        #                      iterations=[self.args.n_iter],
        #                      lamdas=[self.args.lamdas[0], self.args.lamdas[1], self.args.lamdas[2]],
        #                      iteration=["target"],
        #                      target_vertices=self.graph.vertices.reshape(dim, dim) if
        #                      isinstance(self.graph, Grid) else self.graph.vertices,
        #                      target_area_proportions=self.proportion_targets,
        #                      areas=self.graph.areas.reshape(dim, dim) if
        #                      isinstance(self.graph, Grid) else self.graph.areas,
        #                      weights=self.args.weights,
        #                      )


        

            # # plot, print, and log iterations
            # if iteration % self.args.plot_print_and_log_after == 0:
            #     if self.args.plot:
            #         self.plot(iteration=iteration , plots_dir=self.args.plots_dir)

            #     if self.args.print_values:
            #         print("\niteration:", iteration)
            #         print("\nvertices:\n", self.graph.vertices.reshape(dim, dim) if isinstance(self.graph, Grid) else self.graph.vertices)
            #         print("\nmean_vih:\n", self.calc_mean_vih(columns, multipliers, counts).reshape(dim, dim)
            #         if isinstance(self.graph, Grid) else self.calc_mean_vih(columns, multipliers, counts))
            #         print("\ng1 derivative:\n", g1_derivative.reshape(dim, dim) if isinstance(self.graph, Grid) else g1_derivative)
            #         print("\ng2 derivative:\n", g2_derivative.reshape(dim, dim) if isinstance(self.graph, Grid) else g2_derivative)
            #         print("\narea proportions:\n", self.calc_area_proportions(self.args.lamdas[2]))
            #         print("\narea proportion derivative:\n", area_proportion_derivative.reshape(dim, dim) if isinstance(self.graph, Grid) else area_proportion_derivative)
            #         print("\nenergy derivative:\n", energy_derivative.reshape(dim, dim) if isinstance(self.graph, Grid) else energy_derivative)

            #     # Logging
            #     if self.args.log_file:
            #         self.log_to_file(self.args.log_file,
            #                          iteration=[iteration],
            #                          vertices=self.graph.vertices.reshape(dim, dim) if
            #                          isinstance(self.graph, Grid) else self.graph.vertices,
            #                          g1_derivative=g1_derivative.reshape(dim, dim),
            #                          g2_derivative=g2_derivative.reshape(dim, dim),
            #                          area_proportions=self.calc_area_proportions(self.args.lamdas[2]),
            #                          area_proportion_derivative=area_proportion_derivative.reshape(
            #                              dim, dim)
            #                          if isinstance(self.graph, Grid) else area_proportion_derivative,
            #                          energy_derivative=energy_derivative.reshape(
            #                              dim, dim)
            #                          if isinstance(self.graph, Grid) else energy_derivative,
            #                          )


        # # plot, print, and log final result

        # if self.args.plot:
        #     self.plot(iteration='Result'  , plots_dir=self.args.plots_dir)

        # if self.args.print_values:
        #     print("\niteration:", "result")
        #     print("\nvertices:\n", self.graph.vertices.reshape(dim, dim) if isinstance(self.graph, Grid) else self.graph.vertices)
        #     print("\narea_proportions:\n", self.calc_area_proportions(self.args.lamdas[2]))

        # if self.args.log_file:
        #     self.log_to_file(self.args.log_file,
        #                      iteration=["result"],
        #                      vertices=self.graph.vertices.reshape(dim, dim) if
        #                      isinstance(self.graph, Grid) else self.graph.vertices,
        #                      area_proportions = self.calc_area_proportions(self.args.lamdas[2])
        #                      )

    def log_to_file(self, log_file, **kwargs): 
        #rewrite to match results with additional columns for keys that have a value for each vertex, 
        # the remaining keys being stored in a config file along with self.args
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
            raise NotImplementedError("Plotting for graphs is not implemeted. \
                                      Please use log_to_file instead and use GIS software to plot the results")
            # if plots_dir is None:
            #     pd.DataFrame(np.vstack((self.graph.vertices.reshape(1, -1), self.graph.coordinates.T)).T).to_csv(f'iteration: {iteration}.csv')
            # else:
            #     pd.DataFrame(np.vstack((self.graph.vertices.reshape(1, -1), self.graph.coordinates.T)).T).to_csv(f'{plots_dir}/iteration: {iteration}.csv')

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
