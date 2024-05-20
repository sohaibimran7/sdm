#%%
from matplotlib import pyplot as plt
import json
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from dataclasses import dataclass
import pandas as pd

#%% 
FLOODMAP_EVALS = False

if FLOODMAP_EVALS:
    RESULTS_DIR = "results/lancaster/floodmap/"

    EXPERIMENTS = [[f"alpha-{alpha}_train-{train_perc/100}" for train_perc in range(0, 101, 5)] for alpha in (0, 0.1)]

    ALPHA_LABELS = [
    'Rnd dsg',
    'HNN ($\\alpha = 0.1$)'
    ]

else:
    RESULTS_DIR = "results/lancaster"

    EXPERIMENTS = [[f"alpha-{alpha}_train-{train_perc/100}" for train_perc in range(0, 96, 5)] for alpha in (0, 0.1, 0.5, 0.9)]
    
    ALPHA_LABELS = [
    'Rnd dsg',
    '$\\alpha = 0.1$',
    '$\\alpha = 0.5$',
    '$\\alpha = 0.9$'
    ]

experiments_flattened = np.array(EXPERIMENTS).flatten().tolist()

TRAIN_PROPS_LABEL = "Proportion of validation properties seeded"
TRAIN_PROPS = [train_perc/100 for train_perc in range(0, 100, 5)]
train_props_ticks = [str(round(train_prop, 2)) for train_prop in TRAIN_PROPS]

CONFUSION_MATRIX_LABEL = "Proportion of validation properties predicted as 1"

COLORS = [
    'tab:gray', 
    'tab:blue', 
    'tab:orange', 
    'tab:green'
    ]

CLASSIFICATION_METRICS = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score']
PROPORTION_METRICS = ['mean_absolute_error', 'max_error']

CLASSIFICATION_METRIC_LABELS = ['Accuracy', 'F1 score', 'Precision', 'Recall', ]
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
def filter_structure_results(results : dict, configs: dict, experiments : np.array, metrics : list = None, classification=True):
    experiments = np.array(experiments)
    experiments_shape = experiments.shape
    flat_experiments = experiments.flatten()
    if metrics:
        assert not (len(metrics) > 1 and 'confusion_matrix' in metrics), "If confusion matrix is required, it must be the only metric."
        filtered_results = [[[results[experiment][run][0 if classification else 1][metric] for run in range(configs[experiment]['n_runs'])] for experiment in flat_experiments] for metric in metrics]
        original_shape = (len(metrics),) + experiments_shape + np.array(filtered_results).shape[2:]
    else:
        filtered_results = [[results[experiment][run] for run in range(configs[experiment]['n_runs'])] for experiment in flat_experiments]
        original_shape = experiments_shape + np.array(filtered_results).shape[1:]
    return np.array(filtered_results).reshape(original_shape).squeeze()

def get_cm_props_and_ones_ratios(filtered_structured_results):
    conf_mat = filtered_structured_results
    conf_mat_prop = conf_mat / conf_mat.sum(axis=(-1, -2), keepdims=True)
    ones_ratios = conf_mat[..., 1].sum(axis=-1) /  conf_mat.sum(axis = (-1, -2))
    return conf_mat_prop, ones_ratios
#%%
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

def plot_line_with_error(ax, data, x_vals, label=None, color=None, **kwargs):
    """Plot a line with error fill."""
    ax.plot(x_vals, np.mean(data, axis=-1), color=color, label=label)
    ax.fill_between(x_vals, np.min(data, axis=-1), np.max(data, axis=-1), alpha=0.3, color=color)
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
# %%
all_results = {}
for experiment in experiments_flattened:
    with open (f"{RESULTS_DIR}/{experiment}/metrics.json", "r") as f:
        all_results[experiment] = json.load(f)

all_configs = {}
for experiment in experiments_flattened:
    with open (f"{RESULTS_DIR}/{experiment}/config.json", "r") as f:
        all_configs[experiment] = json.load(f)

aoi_ones_ratios = {}
for experiment in experiments_flattened:
    df = pd.read_csv(f"{RESULTS_DIR}/{experiment}/predictions.csv")
    predicted_cols = [col for col in df.columns if col.startswith('predicted_')]
    proportions = [((df[col] == 1).mean()) for col in predicted_cols]
    aoi_ones_ratios[experiment] = proportions
#%%

train_perc = np.array(TRAIN_PROPS) * val_percentage
train_perc_label = "Percentage of properties seeded"

if FLOODMAP_EVALS:
    fig, ax = overlaid_plot(
        data = filter_structure_results(aoi_ones_ratios, all_configs, np.array(EXPERIMENTS)) * 100,
        plot_func = plot_line_with_error,
        x_vals = train_perc,
        xlabel = train_perc_label,
        ylabel = AOI_ONES_YLABEL,
        labels = ALPHA_LABELS,
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
    fig, axs = multi_plot(
            data = filter_structure_results(all_results, all_configs, np.array(EXPERIMENTS), CLASSIFICATION_METRICS, classification=True),
            plot_func = plot_line_with_error,
            grid = [2, 2],
            x_vals = train_perc,
            xlabel = train_perc_label,
            ylabels = CLASSIFICATION_METRIC_LABELS,
            labels = ALPHA_LABELS,
            colors=COLORS
    )
    # save_plot(fig, 'HNN validation classification results')
    plt.show()

    fig, axs = multi_plot(
        data = filter_structure_results(all_results, all_configs, np.array(EXPERIMENTS), PROPORTION_METRICS, classification=False),
        plot_func = plot_line_with_error,
        grid = [1, 2],
        x_vals = train_perc,
        xlabel = train_perc_label,
        ylabels = PROPORTION_METRIC_LABELS,
        labels = ALPHA_LABELS,
        colors=COLORS
    )
    # save_plot(fig, 'HNN validation proportion results')
    plt.show()