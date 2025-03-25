#%%
from matplotlib import pyplot as plt
import json
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd

#%% 
FLOODMAP_EVALS = False

if FLOODMAP_EVALS:
    RESULTS_DIR = "results/lancaster/floodmap"

    EXPERIMENTS = [
        [f"hnn_alpha-{alpha}_train-{train_perc/100}" for train_perc in range(0, 96, 5)] for alpha in (0, 0.1)
    ] + [
        [None if train_perc == 0 else f"k-{k}_train-{train_perc/100}" for train_perc in range(0, 96, 5)] for k in (1, 3)
    ]

    PARAMETER_LABELS = [
        'Rnd dsg',
        'HNN ($\\alpha = 0.1$)',
        'kNN (k=1)',
        'kNN (k=3)',
    ]

else:
    RESULTS_DIR = "results/lancaster"

    EXPERIMENTS = [
        [f"hnn_alpha-{alpha}_train-{train_perc/100}" for train_perc in range(0, 96, 5)] for alpha in (0, 0.1, 0.5, 0.9)
    ] + [
        [None if train_perc == 0 else f"k-{k}_train-{train_perc/100}" for train_perc in range(0, 96, 5)] for k in (1, 3, 5)
    ]

PARAMETER_LABELS = [
    'Rnd dsg',
    'HNN ($\\alpha = 0.1$)',
    'HNN ($\\alpha = 0.5$)',  
    'HNN ($\\alpha = 0.9$)',
    'kNN (k=1)',
    'kNN (k=3)',
    'kNN (k=5)'
]

# Flatten the EXPERIMENTS list manually and filter out None values
experiments_flattened = [experiment for sublist in EXPERIMENTS for experiment in sublist if experiment is not None]

TRAIN_PROPS_LABEL = "Percentage of validation properties seeded"
TRAIN_PROPS = [train_perc for train_perc in range(0, 96, 5)]
train_props_ticks = [str(round(train_prop, 2)) for train_prop in TRAIN_PROPS]

CONFUSION_MATRIX_LABEL = "Proportion of validation properties predicted as 1"

COLORS = [
    'tab:gray',     # Random desaggregation (α=0)
    
    '#4BA6D1',      # α=0.1 (light blue - darkened from #add8e6)
    '#0056a1',      # α=0.5 (darker medium blue)
    '#00008b',      # α=0.9 (dark blue)

    '#FFC300',      # k=1 (bright golden - lighter than previous)
    '#FF9933',      # k=3 (lighter orange - changed from #ff8c00)
    '#FF4D4D',      # k=5 (lighter red - changed from #cc0000)
]

SUMMARY_METRICS = {
    'classification': ['f1_score'],
    'proportion': ['mean_absolute_error']
}

CLASSIFICATION_METRICS = ['accuracy_score', 'matthews_corrcoef', 'cohen_kappa_score', 'f1_score', 'precision_score', 'recall_score']
PROPORTION_METRICS = ['mean_absolute_error', 'max_error']

SUMMARY_METRIC_LABELS = {
    'classification': ['F1 Score'],
    'proportion': ['Mean Absolute Error', 'Max Error']
}

CLASSIFICATION_METRIC_LABELS = ['Accuracy', 'Matthews Correlation Coefficient', 'Cohen\'s Kappa', 'F1 Score', 'Precision', 'Recall']
PROPORTION_METRIC_LABELS = ['Mean Absolute Error', 'Max Error']

AOI_ONES_YLABEL = 'Percentage of properties classified as non-residential'
AOI_ONES_YLABEL_COST = 'Estimated potential flood damage (millions of £)'
RESIDENTIAL_COST = 50000
NON_RESIDENTIAL_COST = 134696

OUTPUT_DIR = "figures/images"

val_labels = pd.read_csv(f"{RESULTS_DIR}/{experiments_flattened[0]}/predictions.csv", usecols=['label']).to_numpy()
val_percentage = 100 * (1 - np.mean(np.isnan(val_labels)))
n_rows = np.sum(val_labels == 1)

#%%
def process_file(experiment, pred_cols, results_dir):
    chunks = pd.read_csv(
        f"{results_dir}/{experiment}/predictions.csv",
        usecols=pred_cols,
        dtype=np.int8,
        engine='c',
        chunksize=10000
    )
    
    running_sum = 0
    total_rows = 0
    for chunk in chunks:
        chunk_sum = (chunk == 1).sum()
        running_sum += chunk_sum
        total_rows += len(chunk)
    
    proportions = running_sum / total_rows
    return experiment, proportions.values

def load_predictions(experiments_flattened, results_dir):
    # Pre-compute column names
    predicted_cols = pd.read_csv(f"{results_dir}/{experiments_flattened[0]}/predictions.csv", nrows=0).columns
    pred_cols = [col for col in predicted_cols if col.startswith('predicted_')]
    
    # Process files in parallel
    aoi_ones_ratios = {}
    process_func = partial(process_file, pred_cols=pred_cols, results_dir=results_dir)
    
    with ProcessPoolExecutor() as executor:
        futures = list(tqdm(
            executor.map(process_func, experiments_flattened),
            total=len(experiments_flattened),
            desc="Loading predictions"
        ))
        
        for experiment, proportions in futures:
            aoi_ones_ratios[experiment] = proportions
            
    return aoi_ones_ratios

def filter_structure_results(results : dict, configs: dict, experiments : np.array, metrics : list = None, classification=True):
    experiments = np.array(experiments)
    experiments_shape = experiments.shape
    flat_experiments = experiments.flatten()
    
    # Find max number of runs across all experiments
    max_runs = max(configs[exp]['n_runs'] for exp in flat_experiments)
    
    if metrics:
        assert not (len(metrics) > 1 and 'confusion_matrix' in metrics), "If confusion matrix is required, it must be the only metric."
        filtered_results = [[[results[experiment][run][0 if classification else 1][metric] 
                            if experiment is not None and run < configs[experiment]['n_runs']
                            else np.nan 
                            for run in range(max_runs)] 
                           for experiment in flat_experiments] 
                          for metric in metrics]
        original_shape = (len(metrics),) + experiments_shape + (max_runs,)
    else:
        filtered_results = [[results[experiment][run] 
                           if experiment is not None and run < configs[experiment]['n_runs']
                           else np.nan 
                           for run in range(max_runs)] 
                          for experiment in flat_experiments]
        original_shape = experiments_shape + (max_runs,)
    
    return np.array(filtered_results).reshape(original_shape).squeeze()

def get_cm_props_and_ones_ratios(filtered_structured_results):
    conf_mat = filtered_structured_results
    conf_mat_prop = conf_mat / conf_mat.sum(axis=(-1, -2), keepdims=True)
    ones_ratios = conf_mat[..., 1].sum(axis=-1) /  conf_mat.sum(axis = (-1, -2))
    return conf_mat_prop, ones_ratios

def plot_violin(ax, data, x_vals, **kwargs):
    """Plot a violin plot."""
    ax.violinplot(data.T, showmeans=True)
    ax.set_xticks(np.arange(1, len(x_vals) + 1))
    ax.set_xticklabels(x_vals)
    ax.set(**kwargs)

def plot_line(ax, data, x_vals, label=None, color=None, **kwargs):
    """Plot a line."""
    ax.plot(x_vals, np.mean(data, axis=-1), color=color, label=label)
    ax.set(**kwargs)

def plot_line_with_error(ax, data, x_vals, label=None, color=None, opacity=0.2, 
                        uncertainty_type='std', uncertainty_param=1.0, **kwargs):
    """Plot a line with error fill.
    
    Parameters:
    -----------
    uncertainty_type : str
        Type of uncertainty measure to use ('range', 'std', or 'ci')
    uncertainty_param : float
        Parameter for uncertainty calculation:
        - For 'std': number of standard deviations
        - For 'ci': confidence level (e.g., 0.95 for 95% CI)
        - For 'range': ignored (uses min/max)
    """
    mean = np.mean(data, axis=-1)
    ax.plot(x_vals, mean, color=color, label=label)
    
    if uncertainty_type == 'range':
        lower = np.min(data, axis=-1)
        upper = np.max(data, axis=-1)
    elif uncertainty_type == 'std':
        std = np.std(data, axis=-1)
        lower = mean - uncertainty_param * std
        upper = mean + uncertainty_param * std
    elif uncertainty_type == 'ci':
        from scipy import stats
        ci = stats.t.interval(
            uncertainty_param, 
            df=data.shape[-1]-1,
            loc=mean,
            scale=stats.sem(data, axis=-1)
        )
        lower, upper = ci
    else:
        raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")
    
    ax.fill_between(x_vals, lower, upper, alpha=opacity, color=color)
    kwargs.pop('color', None)
    ax.set(**kwargs)

def plot_confusion_matrix(ax, data, x_vals=None, **kwargs):
    return ConfusionMatrixDisplay(data).plot(ax=ax, cmap='Blues', **kwargs)

def plot_bar(ax, data, x_vals, **kwargs):
    ax.bar(x_vals, data)
    ax.set(**kwargs)

@dataclass
class HLine:
    y: float = np.nanmean(val_labels)
    color: str = 'grey'
    linestyle: str = '--'
    text: str = 'Proportion of 1s\nin validation data'
    text_x: float = 5
    text_verticalalignment: str = 'center'
    text_horizontalalignment: str = 'left'

def plot_clustered_bar(ax, data_groups, group_labels, bar_labels, hlines: list[HLine], **kwargs):
    n_groups = len(data_groups)
    n_bars = len(data_groups[0])
    bar_width = 0.8 / n_bars  # Width of bars to fit all bars within each cluster
    index = np.arange(n_groups)

    for i, data in enumerate(zip(*data_groups)):  # Transpose to iterate over bars within groups
        ax.bar(index + i * bar_width, data, bar_width, label=bar_labels[i])
    
    for hline in hlines:
        ax.axhline(y=hline.y, color=hline.color, linestyle=hline.linestyle)
        ax.text(x=hline.text_x, y=hline.y, s=hline.text, verticalalignment=hline.text_verticalalignment, horizontalalignment=hline.text_horizontalalignment)

    ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(group_labels)
    ax.set(**kwargs)
    ax.legend()

# %%
def overlaid_plot(data : np.array, plot_func, ax = None, x_vals=None, title=None, ylabel=None, labels=None, colors=None, **kwargs):
    assert data.ndim < 4, "data must be at most 4D"
    if not ax:
        fig, ax = plt.subplots()
    for i in range(len(data)):
        plot_func(ax, data[i],
                    x_vals = x_vals,
                    label = labels[i],
                    color = colors[i] if colors else None,
                    **kwargs)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
    ax.legend()
    return fig, ax

def multi_plot(data : np.array, plot_func, grid=None, x_vals=None, main_title=None, titles=None, ylabels=None, labels=None, axs=None, fig=None, colors=None, **kwargs):
    """Create a grid of subplots."""
    assert data.ndim < 5, "data must be at most 4D"
    if axs is None:
        assert grid is not None, "If axs is not provided, grid must be provided"
        assert np.prod(grid) >= len(data), "Grid size must match or exceed data to plot"
        fig, axs = plt.subplots(*grid, figsize=(grid[1] * 5, grid[0] * 5))
        axs = np.atleast_1d(axs).flatten()  # Ensure axs is always iterable
        fig.suptitle(main_title)
    for i in range(len(data)):
        if data[i].ndim < 3:
            plot_func(axs[i],
                      data[i],
                      x_vals=x_vals,
                      **kwargs)
            if ylabels:
                axs[i].set_ylabel(ylabels[i])
            if titles:
                axs[i].set_title(titles[i])
        else:
            for j in range(len(data[i])):
                plot_func(axs[i], data[i, j],
                          x_vals=x_vals,
                          label = labels[j],
                          color = colors[j] if colors else None,
                          **kwargs)
                if ylabels:
                    axs[i].set_ylabel(ylabels[i])
                if titles:
                    axs[i].set_title(titles[i])
            handles, labels = axs[i].get_legend_handles_labels()
            filtered_handles = [h for h, l in zip(handles, labels) if l is not None and l != '']
            axs[i].legend(filtered_handles, labels)    
                
    plt.tight_layout(w_pad=3, h_pad=2.5)
    return fig, axs 

def save_plot(fig, name):
    fig.savefig(f"{OUTPUT_DIR}/{name}.jpeg", format='jpeg', dpi=300)

def plot_pareto(data_x, data_y, ax, use_mean=True, labels=PARAMETER_LABELS, colors=COLORS, hnn_indices=slice(1, 4), knn_indices=slice(4, 7), **kwargs):
    """
    Plot values of any two metrics against each other.
    
    Parameters:
    -----------
    data_x : np.array
        Data for x-axis metric
    data_y : np.array
        Data for y-axis metric
    ax : matplotlib.axes.Axes
        Axes to plot on
    use_mean : bool
        If True, plot mean values. If False, plot all runs as separate points
    labels : list
        Labels for each point
    colors : list
        Colors for each point
    kwargs : dict
        Additional keyword arguments for ax.set()
    """
    if use_mean:
        x_vals = np.mean(data_x, axis=-1)
        y_vals = np.mean(data_y, axis=-1)
    else:
        x_vals = data_x
        y_vals = data_y
    
    # Plot random disaggregation point
    if use_mean:
        ax.scatter(x_vals[0], y_vals[0], label=labels[0], color=colors[0])
    else:
        ax.scatter(x_vals[0], y_vals[0], label=labels[0], color=colors[0], alpha=0.5)
    
    if use_mean:
        ax.plot(x_vals[hnn_indices], y_vals[hnn_indices], 
                color='blue', linestyle='--', alpha=0.5)
        for i, (x, y) in enumerate(zip(x_vals[hnn_indices], y_vals[hnn_indices])):
            ax.scatter(x, y, color=colors[i+1], label=f'{labels[i+1]}')
    else:
        for i in range(3):  # for each alpha value
            idx = i + 1
            ax.scatter(x_vals[idx], y_vals[idx], 
                      color=colors[idx], label=f'{labels[idx]}', alpha=0.5)
    
    # Plot kNN points
    if use_mean:
        ax.plot(x_vals[knn_indices], y_vals[knn_indices], 
                color='red', linestyle='--', alpha=0.5)
        for i, (x, y) in enumerate(zip(x_vals[knn_indices], y_vals[knn_indices])):
            ax.scatter(x, y, color=colors[i+knn_indices.start], label=labels[i+knn_indices.start])
    else:
        for i in range(3):  # for each k value
            idx = i + knn_indices.start
            ax.scatter(x_vals[idx], y_vals[idx], 
                      color=colors[idx], label=labels[idx], alpha=0.5)
    
    ax.set(**kwargs)

def create_pareto_plot(metric_x, metric_y, x_label, y_label, use_mean=True, invert_x=False, 
                      classification_x=False, classification_y=True, selected_percentages = [5, 20, 50, 80, 95]):
    """
    Create Pareto plot for any two metrics.
    
    Parameters:
    -----------
    metric_x : str
        Name of metric for x-axis
    metric_y : str
        Name of metric for y-axis
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    use_mean : bool
        If True, plot mean values. If False, plot all runs as separate points
    invert_x : bool
        Whether to invert x-axis
    classification_x : bool
        Whether x-axis metric is a classification metric
    classification_y : bool
        Whether y-axis metric is a classification metric
    """
    prop_indices = [TRAIN_PROPS.index(prop) for prop in selected_percentages]

    # Get metrics for all experiments
    x_scores = filter_structure_results(all_results, all_configs, np.array(EXPERIMENTS), 
                                      [metric_x], classification=classification_x)
    y_scores = filter_structure_results(all_results, all_configs, np.array(EXPERIMENTS), 
                                      [metric_y], classification=classification_y)

    # Create subplots for selected proportions
    fig, axs = plt.subplots(1, len(selected_percentages), figsize=(20, 4))
    for idx, (ax, prop_idx) in enumerate(zip(axs, prop_indices)):
        plot_pareto(
            data_x=x_scores[:, prop_idx],
            data_y=y_scores[:, prop_idx],
            ax=ax,
            use_mean=use_mean,
            labels=PARAMETER_LABELS,
            colors=COLORS,
            xlabel=x_label,
            ylabel=y_label if idx == 0 else '',
            title=f'Percentage seeded: {selected_percentages[idx]}%'
        )
        if invert_x:
            ax.invert_xaxis()
        if idx == len(selected_percentages)-1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig, axs

#%%
if __name__ == "__main__":

    all_results = {}
    for experiment in experiments_flattened:
        with open (f"{RESULTS_DIR}/{experiment}/metrics.json", "r") as f:
            all_results[experiment] = json.load(f)

    all_configs = {}
    for experiment in experiments_flattened:
        with open (f"{RESULTS_DIR}/{experiment}/config.json", "r") as f:
            all_configs[experiment] = json.load(f)

    train_perc = np.array(TRAIN_PROPS) * val_percentage
    train_perc_label = "Percentage of properties seeded"

    if FLOODMAP_EVALS:

        aoi_ones_ratios = load_predictions(experiments_flattened, RESULTS_DIR)

        fig, ax = overlaid_plot(
            data = filter_structure_results(aoi_ones_ratios, all_configs, np.array(EXPERIMENTS)) * 100,
            plot_func = plot_line_with_error,
            x_vals = train_perc,
            xlabel = train_perc_label,
            ylabel = AOI_ONES_YLABEL,
            labels = PARAMETER_LABELS,
            colors = COLORS
        )

        axr = ax.twinx()
        axr.set_ylabel(AOI_ONES_YLABEL_COST)
        axr.set_ylim((RESIDENTIAL_COST + ax.get_ylim()[0]/100 * (NON_RESIDENTIAL_COST - RESIDENTIAL_COST)) * len(val_labels)/1000000, 
                    (RESIDENTIAL_COST + ax.get_ylim()[1]/100 * (NON_RESIDENTIAL_COST - RESIDENTIAL_COST)) * len(val_labels)/1000000
                    )

        # save_plot(fig, 'flood damage and ones percent')
        plt.show()

    else:
        # Add new summary metrics plot
        fig, axs = multi_plot(
            data = np.array([
                filter_structure_results(all_results, all_configs, np.array(EXPERIMENTS), SUMMARY_METRICS['classification'], classification=True),
                filter_structure_results(all_results, all_configs, np.array(EXPERIMENTS), SUMMARY_METRICS['proportion'], classification=False)
            ]),
            plot_func = plot_line_with_error,
            grid = [1, 2],
            x_vals = TRAIN_PROPS,
            xlabel = TRAIN_PROPS_LABEL,
            ylabels = [label[0] for label in SUMMARY_METRIC_LABELS.values()],
            labels = PARAMETER_LABELS,
            colors = COLORS
        )
        save_plot(fig, 'validation summary results')
        plt.show()

        # Existing classification metrics plot
        fig, axs = multi_plot(
            data = filter_structure_results(all_results, all_configs, np.array(EXPERIMENTS), CLASSIFICATION_METRICS, classification=True),
            plot_func = plot_line_with_error,
            grid = [2, 3],
            x_vals = TRAIN_PROPS,
            xlabel = TRAIN_PROPS_LABEL,
            ylabels = CLASSIFICATION_METRIC_LABELS,
            labels = PARAMETER_LABELS,
            colors=COLORS
        )
        save_plot(fig, 'validation classification results')
        plt.show()

        fig, axs = multi_plot(
            data = filter_structure_results(all_results, all_configs, np.array(EXPERIMENTS), PROPORTION_METRICS, classification=False),
            plot_func = plot_line_with_error,
            grid = [1, 2],
            x_vals = TRAIN_PROPS,
            xlabel = TRAIN_PROPS_LABEL,
            ylabels = PROPORTION_METRIC_LABELS,
            labels = PARAMETER_LABELS,
            colors=COLORS
        )
        save_plot(fig, 'validation proportion results')
        plt.show()

        metric_x = 'mean_absolute_error'
        metric_x_label = 'Mean Absolute Error'
        metric_y = 'f1_score'
        metric_y_label = 'F1 Score'

        fig, axs = create_pareto_plot(
            metric_x=metric_x,
            metric_y=metric_y,
            x_label=metric_x_label,
            y_label=metric_y_label,
            use_mean=True,
            invert_x=True,
            classification_x=False,
            classification_y=True
        )
        plt.show()
        save_plot(fig, 'validation pareto plot')

# %%
