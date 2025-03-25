# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, mean_absolute_error, max_error
from pathlib import Path
from tqdm import tqdm
import re
from typing import List, Dict, Tuple, Optional, Union, Any

# Configuration
RESULTS_DIR = "results/lancaster/test"
OUTPUT_DIR = "figures/images"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_PERCENTAGES = range(0, 40, 10)
ALPHAS = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


EXPERIMENTS = [
    [f"hnn_alpha-{alpha}_train-{train_perc / 100}" for train_perc in TRAIN_PERCENTAGES]
    for alpha in ALPHAS
]

PARAMETER_LABELS = [
    "Rnd dsg",
    "HNN ($\\alpha = 0.001$)",
    "HNN ($\\alpha = 0.005$)",
    "HNN ($\\alpha = 0.01$)",
    "HNN ($\\alpha = 0.05$)",
    "HNN ($\\alpha = 0.1$)",
    "HNN ($\\alpha = 0.5$)",
]

COLORS = [
    "#808080",  # Random desaggregation (α=0) - grey
    "#deebf7",  # α=0.001 - very light blue
    "#9ecae1",  # α=0.005 - light blue
    "#6baed6",  # α=0.01 - medium blue
    "#3182bd",  # α=0.05 - dark blue
    "#08519c",  # α=0.1 - very dark blue
    "#08306b",  # α=0.5 - deepest blue
]

TRAIN_PROPS_LABEL = "Percentage of validation properties seeded"
TRAIN_PROPS = TRAIN_PERCENTAGES

# Define metrics
CLASSIFICATION_METRICS = {
    'accuracy': accuracy_score,
    'matthews_corrcoef': matthews_corrcoef,
    'cohen_kappa': cohen_kappa_score,
    'f1': f1_score,
    'precision': precision_score,
    'recall': recall_score
}

PROPORTION_METRICS = {
    'mae': mean_absolute_error,
    'max_error': max_error
}

METRIC_LABELS = {
    'accuracy': 'Accuracy',
    'matthews_corrcoef': 'Matthews Correlation Coefficient',
    'cohen_kappa': "Cohen's Kappa",
    'f1': 'F1 Score',
    'precision': 'Precision',
    'recall': 'Recall',
    'mae': 'Mean Absolute Error',
    'max_error': 'Max Error'
}

# Flatten experiments
experiments_flattened = [exp for sublist in EXPERIMENTS for exp in sublist]


def extract_experiment_params(experiment_name: str) -> Tuple[float, float]:
    """Extract alpha and train percentage from experiment name"""
    alpha_match = re.search(r'alpha-(\d+\.\d+|\d+)', experiment_name)
    train_match = re.search(r'train-(\d+\.\d+)', experiment_name)
    
    alpha = float(alpha_match.group(1)) if alpha_match else 0
    train_perc = float(train_match.group(1)) * 100 if train_match else 0
    
    return alpha, train_perc


def load_predictions() -> List[List[pd.DataFrame]]:
    """
    Load all prediction data into a nested list structure that mirrors EXPERIMENTS.
    
    Returns:
        A list of lists where:
        - The outer list corresponds to different alpha values (0, 0.01, 0.05)
        - Each inner list contains DataFrames for different train_percentages
    """
    # Initialize the structure to match EXPERIMENTS
    predictions_by_params = [[] for _ in range(len(EXPERIMENTS))]
    
    # Create a mapping from experiment name to its position in the structure
    experiment_to_position = {}
    for i, alpha_list in enumerate(EXPERIMENTS):
        for j, experiment in enumerate(alpha_list):
            experiment_to_position[experiment] = (i, j)
    
    # Load each experiment
    for experiment in tqdm(experiments_flattened, desc="Loading predictions"):
        try:
            # Load experiment file
            file_path = f"{RESULTS_DIR}/{experiment}/predictions.csv"
            df = pd.read_csv(file_path)
            
            # Extract parameters
            alpha, train_perc = extract_experiment_params(experiment)
            
            # Add experiment metadata to the DataFrame for easier analysis
            df['experiment'] = experiment
            df['alpha'] = alpha
            df['train_percentage'] = train_perc
            
            # Get the position in the structure
            if experiment in experiment_to_position:
                i, j = experiment_to_position[experiment]
                
                # Ensure the inner list has enough slots
                while len(predictions_by_params[i]) <= j:
                    predictions_by_params[i].append(None)
                
                # Store the DataFrame
                predictions_by_params[i][j] = df
            
        except Exception as e:
            print(f"Error loading {experiment}: {e}")
    
    return predictions_by_params


def calculate_metrics(predictions_by_params: List[List[pd.DataFrame]]) -> pd.DataFrame:
    """Calculate metrics from the predictions and return as a DataFrame"""
    rows = []
    
    # Iterate through alphas
    for alpha_idx, alpha_list in enumerate(predictions_by_params):
        # Iterate through training percentages
        for train_idx, df in enumerate(alpha_list):
            if df is None:
                continue
                
            # Extract parameters
            alpha = df['alpha'].iloc[0]
            train_perc = df['train_percentage'].iloc[0]
            experiment = df['experiment'].iloc[0]
            
            # Find run columns
            run_columns = [col for col in df.columns if col.startswith('predicted_')]
            n_runs = len(run_columns)
            
            # Process each run
            for run_idx in range(1, n_runs + 1):
                pred_col = f'predicted_{run_idx}'
                train_col = f'train_{run_idx}'
                
                # Skip if columns don't exist
                if pred_col not in df.columns or train_col not in df.columns:
                    continue
                
                # Classification metrics (on non-seeded validation points)
                valid_mask = ~df['label'].isna()
                not_seeded_mask = df[train_col].isna()
                eval_mask = valid_mask & not_seeded_mask
                
                class_metrics = {}
                if eval_mask.sum() > 0:
                    y_true = df.loc[eval_mask, 'label'].astype(int)
                    y_pred = df.loc[eval_mask, pred_col].astype(int)
                    
                    for metric_name, metric_func in CLASSIFICATION_METRICS.items():
                        try:
                            class_metrics[metric_name] = metric_func(y_true, y_pred)
                        except Exception:
                            class_metrics[metric_name] = np.nan
                
                # Proportion metrics (area-level)
                prop_metrics = {}
                if 'area_id' in df.columns:
                    area_groups = df.groupby('area_id')
                    
                    area_true_props = []
                    area_pred_props = []
                    
                    for area_id, area_df in area_groups:
                        valid_area_mask = ~area_df['label'].isna()
                        if valid_area_mask.sum() > 0:
                            true_prop = area_df.loc[valid_area_mask, 'label'].mean()
                            pred_prop = area_df[pred_col].mean()
                            
                            area_true_props.append(true_prop)
                            area_pred_props.append(pred_prop)
                    
                    for metric_name, metric_func in PROPORTION_METRICS.items():
                        try:
                            prop_metrics[metric_name] = metric_func(area_true_props, area_pred_props)
                        except Exception:
                            prop_metrics[metric_name] = np.nan
                
                # Create a row with all information
                row = {
                    'experiment': experiment,
                    'alpha': alpha,
                    'train_percentage': train_perc,
                    'run': run_idx,
                    **class_metrics,
                    **prop_metrics
                }
                
                rows.append(row)
    
    return pd.DataFrame(rows)


def combine_predictions(predictions_by_params: List[List[pd.DataFrame]]) -> pd.DataFrame:
    """
    Combine all predictions into a single DataFrame.
    This is useful for some types of exploratory analysis.
    """
    all_dfs = []
    
    for alpha_list in predictions_by_params:
        for df in alpha_list:
            if df is not None:
                # Make a copy to avoid modifying the original
                all_dfs.append(df.copy())
    
    # Combine all DataFrames
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def plot_metrics(results_df: pd.DataFrame, metric_type: str, metric_names: List[str]):
    """Plot metrics from the results DataFrame"""
    # Create plots for each metric
    for metric_name in metric_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by alpha and calculate statistics
        for i, alpha in enumerate(sorted(results_df['alpha'].unique())):
            alpha_df = results_df[results_df['alpha'] == alpha]
            
            # Group by train percentage and calculate mean and std
            stats = alpha_df.groupby('train_percentage')[metric_name].agg(['mean', 'std']).reset_index()
            
            # Plot line with error bands
            color = COLORS[i] if i < len(COLORS) else None
            alpha_idx = [0, 0.01, 0.05].index(alpha) if alpha in [0, 0.01, 0.05] else -1
            label = PARAMETER_LABELS[alpha_idx] if alpha_idx >= 0 else f"α={alpha}"
            
            ax.plot(stats['train_percentage'], stats['mean'], label=label, color=color)
            ax.fill_between(
                stats['train_percentage'], 
                stats['mean'] - stats['std'],
                stats['mean'] + stats['std'],
                alpha=0.2, 
                color=color
            )
        
        # Set labels and title
        ax.set_xlabel(TRAIN_PROPS_LABEL)
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name.title()))
        ax.set_title(f'{METRIC_LABELS.get(metric_name, metric_name.title())}')
        ax.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{metric_name}_plot.jpeg", format="jpeg", dpi=300)
        
        # Display the plot interactively
        plt.show()
        plt.close()


def create_summary_plot(results_df: pd.DataFrame):
    """Create a summary plot with F1 score and MAE using the DataFrame structure"""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    metrics = ['f1', 'mae']
    titles = ['F1 Score (Classification)', 'Mean Absolute Error (Proportion)']
    
    for i, (metric_name, title) in enumerate(zip(metrics, titles)):
        ax = axs[i]
        
        # Group by alpha and calculate statistics
        for j, alpha in enumerate(sorted(results_df['alpha'].unique())):
            alpha_df = results_df[results_df['alpha'] == alpha]
            
            # Group by train percentage and calculate mean and std
            stats = alpha_df.groupby('train_percentage')[metric_name].agg(['mean', 'std']).reset_index()
            
            # Plot line with error bands
            color = COLORS[j] if j < len(COLORS) else None
            alpha_idx = [0, 0.01, 0.05].index(alpha) if alpha in [0, 0.01, 0.05] else -1
            label = PARAMETER_LABELS[alpha_idx] if alpha_idx >= 0 else f"α={alpha}"
            
            ax.plot(stats['train_percentage'], stats['mean'], label=label, color=color)
            ax.fill_between(
                stats['train_percentage'], 
                stats['mean'] - stats['std'],
                stats['mean'] + stats['std'],
                alpha=0.2, 
                color=color
            )
        
        # Set labels
        ax.set_xlabel(TRAIN_PROPS_LABEL)
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name.title()))
        ax.set_title(title)
    
    # Use one legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Make room for the legend
    plt.savefig(f"{OUTPUT_DIR}/summary_plot.jpeg", format="jpeg", dpi=300)
    
    # Display the plot interactively
    plt.show()
    plt.close()


def create_pareto_plot(results_df: pd.DataFrame, selected_percentages=[0, 10, 20, 30]):
    """Create Pareto plots for selected training percentages using the DataFrame structure"""
    fig, axs = plt.subplots(1, len(selected_percentages), figsize=(20, 5))
    
    for i, train_perc in enumerate(selected_percentages):
        ax = axs[i]
        
        # Filter data for this training percentage
        perc_df = results_df[results_df['train_percentage'] == train_perc]
        
        # Group by alpha and calculate mean metrics
        for j, alpha in enumerate(sorted(perc_df['alpha'].unique())):
            alpha_df = perc_df[perc_df['alpha'] == alpha]
            
            # Calculate mean metrics for this alpha and train percentage
            mae = alpha_df['mae'].mean()
            f1 = alpha_df['f1'].mean()
            
            # Skip if metrics are missing
            if np.isnan(mae) or np.isnan(f1):
                continue
                
            # Determine label and color
            alpha_idx = [0, 0.01, 0.05].index(alpha) if alpha in [0, 0.01, 0.05] else -1
            label = PARAMETER_LABELS[alpha_idx] if alpha_idx >= 0 else f"α={alpha}"
            color = COLORS[alpha_idx] if alpha_idx >= 0 else None
            
            ax.scatter(mae, f1, label=label, color=color, s=100)
        
        # Set up axis labels and title
        ax.set_xlabel('Mean Absolute Error')
        if i == 0:
            ax.set_ylabel('F1 Score')
        ax.set_title(f'Training: {train_perc}%')
        
        # Invert x-axis (lower MAE is better)
        ax.invert_xaxis()
        
        # Add legend for the last plot
        if i == len(selected_percentages) - 1:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax.legend(unique_handles, unique_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pareto_plot.jpeg", format="jpeg", dpi=300)
    
    # Display the plot interactively
    plt.show()
    plt.close()


def create_proportion_comparison_df(predictions_by_params: List[List[pd.DataFrame]]):
    """
    Create a DataFrame comparing target proportions against predicted proportions for each area.
    
    Args:
        predictions_by_params: Nested list of prediction DataFrames
    
    Returns:
        DataFrame with area-level target and predicted proportions, plus value counts
    """
    # Get the first dataset from each alpha group (assuming train_perc=0)
    alpha_dfs = [alpha[0] for alpha in predictions_by_params if len(alpha) > 0 and alpha[0] is not None]
    
    if not alpha_dfs:
        print("No valid prediction data found")
        return pd.DataFrame()
    
    # Initialize results DataFrame
    results = []
    
    # Process each area
    area_ids = sorted(alpha_dfs[0]['area_id'].unique())
    
    for area_id in area_ids:
        row = {'area_id': area_id}
        
        # Calculate target proportion from the first DataFrame (should be the same in all)
        area_df = alpha_dfs[0][alpha_dfs[0]['area_id'] == area_id]
        valid_mask = ~area_df['label'].isna()
        
        if valid_mask.sum() > 0:
            # Target proportion
            target_prop = area_df.loc[valid_mask, 'label'].mean()
            row['target_proportion'] = target_prop
            
            # Value counts for target
            target_counts = area_df.loc[valid_mask, 'label'].value_counts()
            row['target_count_0'] = target_counts.get(0, 0)
            row['target_count_1'] = target_counts.get(1, 0)
            row['target_total'] = valid_mask.sum()
        
        # Calculate predicted proportions for each alpha
        for i, df in enumerate(alpha_dfs):
            alpha = df['alpha'].iloc[0]
            area_df = df[df['area_id'] == area_id]
            
            # Find run columns
            run_columns = [col for col in area_df.columns if col.startswith('predicted_')]
            
            # Calculate average proportion across all runs
            for run_col in run_columns:
                pred_prop = area_df[run_col].mean()
                run_num = run_col.split('_')[1]
                row[f'alpha_{alpha}_run_{run_num}_proportion'] = pred_prop
                
                # Value counts for predictions
                pred_counts = area_df[run_col].value_counts()
                row[f'alpha_{alpha}_run_{run_num}_count_0'] = pred_counts.get(0, 0)
                row[f'alpha_{alpha}_run_{run_num}_count_1'] = pred_counts.get(1, 0)
            
        results.append(row)
    
    return pd.DataFrame(results)


def plot_proportion_comparisons(proportion_comparison_df: pd.DataFrame, alphas: List[float] = ALPHAS):
    """
    Create visualizations for comparing target vs predicted proportions across areas.
    
    Args:
        proportion_comparison_df: DataFrame with area-level proportions
    """
    n_alphas = len(alphas)

    # Calculate global min and max for consistent x-axis limits
    all_target_props = proportion_comparison_df['target_proportion']
    all_pred_props = []
    for alpha in alphas:
        alpha_str = str(float(alpha))
        pred_col = f'alpha_{alpha_str}_run_1_proportion'
        if pred_col in proportion_comparison_df.columns:
            all_pred_props.extend(proportion_comparison_df[pred_col])

    if all_pred_props:  # Ensure we have prediction data before calculating min/max
        min_val = min(all_target_props.min(), min(all_pred_props))
        max_val = max(all_target_props.max(), max(all_pred_props))
    else:
        min_val = all_target_props.min()
        max_val = all_target_props.max()
        print("Warning: No prediction data found. Using target proportion range for limits.")

    # Add a small buffer to the limits
    buffer = 0.05 * (max_val - min_val)
    min_val -= buffer
    max_val += buffer

    # 1. Scatter plots comparing target vs predicted proportions for each alpha
    fig, axes = plt.subplots(1, n_alphas, figsize=(18, 6))
    
    # For legend: create size reference points
    size_reference_values = []
    size_reference_labels = []
    
    for i, alpha in enumerate(alphas):
        ax = axes[i]
        
        # Find columns for this alpha (using first run)
        alpha_str = str(float(alpha))
        pred_col = f'alpha_{alpha_str}_run_1_proportion'
        count_col_0 = f'alpha_{alpha_str}_run_1_count_0'
        count_col_1 = f'alpha_{alpha_str}_run_1_count_1'
        
        if pred_col not in proportion_comparison_df.columns:
            print(f"Warning: Column {pred_col} not found in DataFrame")
            continue
        
        # Calculate total count for each area (for sizing the dots)
        total_count = proportion_comparison_df[count_col_0] + proportion_comparison_df[count_col_1]
        
        # Scale the sizes to be more visible - no square root scaling
        size_scale = 50 * (total_count / total_count.max())
            
        # Create scatter plot with size based on total count - using unfilled black circles
        ax.scatter(
            proportion_comparison_df['target_proportion'], 
            proportion_comparison_df[pred_col],
            alpha=0.7,
            s=size_scale,  # Size proportional to total count
            facecolors='none',  # Unfilled circles
            edgecolors='black',  # Black edges
            linewidths=1  # Edge width
        )
        
        # Add diagonal line (perfect prediction)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate correlation
        corr = proportion_comparison_df['target_proportion'].corr(proportion_comparison_df[pred_col])
        
        # Add labels and title
        ax.set_xlabel('Target Proportion')
        ax.set_ylabel('Predicted Proportion')
        ax.set_title(f'α = {alpha} (r = {corr:.3f})')
        
        # Set equal aspect ratio and axis limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_val, max_val)  # Set consistent x-axis limits
        ax.set_ylim(min_val, max_val)  # Set consistent y-axis limits (for scatter plot)
        
        # For the first plot, create size reference for legend
        if i == 0:
            # Get min, median, and max counts for reference
            min_count = total_count.min()
            median_count = total_count.median()
            max_count = total_count.max()
            
            # Store values for legend
            size_reference_values = [min_count, median_count, max_count]
            size_reference_labels = [f"Min: {int(min_count)}", 
                                    f"Median: {int(median_count)}", 
                                    f"Max: {int(max_count)}"]
    
    # Add size legend to the last subplot
    if size_reference_values:
        # Calculate sizes for legend
        legend_sizes = 50 * (np.array(size_reference_values) / size_reference_values[-1])
        
        # Create a separate legend for sizes - using unfilled black circles
        legend_elements = []
        for size, label in zip(legend_sizes, size_reference_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             label=label, markerfacecolor='none', 
                                             markeredgecolor='black', markersize=np.sqrt(size/np.pi)))
        
        # Add the legend to the last subplot
        if n_alphas > 0:
            axes[-1].legend(handles=legend_elements, title="Area Size (points)", 
                           loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/proportion_scatter_comparison.jpeg", format="jpeg", dpi=300)

    plt.show()
    plt.close()
    
    # 2. Error distribution (histogram of differences)
    fig, axes = plt.subplots(1, n_alphas, figsize=(20, 6))  # Increased width

    # Calculate error limits for consistent x-axis
    all_errors = []
    for alpha in alphas:
        alpha_str = str(float(alpha))
        pred_col = f'alpha_{alpha_str}_run_1_proportion'
        if pred_col in proportion_comparison_df.columns:
            errors = proportion_comparison_df[pred_col] - proportion_comparison_df['target_proportion']
            all_errors.extend(errors)

    if all_errors:
        min_error = min(all_errors)
        max_error = max(all_errors)
    else:  # Handle case where no errors are calculated
        min_error = -0.1
        max_error = 0.1
        print("Warning: No error data. Using default error range.")

    error_buffer = 0.05 * (max_error - min_error)
    min_error -= error_buffer
    max_error += error_buffer
    
    for i, alpha in enumerate(alphas):
        ax = axes[i]
        
        # Find columns for this alpha (using first run)
        alpha_str = str(float(alpha))
        pred_col = f'alpha_{alpha_str}_run_1_proportion'
        
        if pred_col not in proportion_comparison_df.columns:
            continue
            
        # Calculate errors
        errors = proportion_comparison_df[pred_col] - proportion_comparison_df['target_proportion']
        
        # Create histogram
        ax.hist(errors, bins=30, alpha=0.7, range=(min_error, max_error)) # Set range for consistent x-axis
        
        # Add vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--')
        
        # Calculate mean and std of errors
        mean_error = errors.mean()
        std_error = errors.std()
        
        # Add labels and title with smaller font and better formatting
        ax.set_xlabel('Prediction Error (Predicted - Target)')
        ax.set_ylabel('Count')
        ax.set_title(f'α = {alpha}\nμ = {mean_error:.3f}, σ = {std_error:.3f}', fontsize=10)

        ax.set_xlim(min_error, max_error) # Set consistent x-axis limits
    
    plt.tight_layout(pad=1.5)  # Increased padding between subplots
    plt.savefig(f"{OUTPUT_DIR}/proportion_error_distribution.jpeg", format="jpeg", dpi=300)
    plt.show()
    plt.close()
    
    # 3. Scatter plot of absolute errors with size representing area size
    plt.figure(figsize=(12, 8))
    
    # For legend: create size reference points
    size_reference_values = []
    size_reference_labels = []
    
    # Prepare data for all alphas
    for j, alpha in enumerate(alphas):
        alpha_str = str(float(alpha))
        pred_col = f'alpha_{alpha_str}_run_1_proportion'
        count_col_0 = f'alpha_{alpha_str}_run_1_count_0'
        count_col_1 = f'alpha_{alpha_str}_run_1_count_1'
        
        if pred_col in proportion_comparison_df.columns:
            # Calculate absolute error
            abs_errors = abs(proportion_comparison_df[pred_col] - proportion_comparison_df['target_proportion'])
            
            # Calculate total count for sizing
            if count_col_0 in proportion_comparison_df.columns and count_col_1 in proportion_comparison_df.columns:
                total_count = proportion_comparison_df[count_col_0] + proportion_comparison_df[count_col_1]
            else:
                total_count = proportion_comparison_df['target_total']
            
            # Scale sizes - no square root scaling
            size_scale = 100 * (total_count / total_count.max())
            
            # Add jitter to x-position to avoid overlap
            x_pos = j + 0.1 * (np.random.random(len(abs_errors)) - 0.5)
            
            # Plot scatter with size based on total count - using unfilled black circles
            plt.scatter(x_pos, abs_errors, s=size_scale, alpha=0.7, 
                       facecolors='none', edgecolors='black', linewidths=1)
            
            # Calculate and plot mean (weighted by area size)
            weighted_mean = np.average(abs_errors, weights=total_count)
            plt.plot([j-0.3, j+0.3], [weighted_mean, weighted_mean], 
                     color='black', linewidth=3)
            
            # Add text label with the weighted mean value
            plt.text(j, weighted_mean + 0.01, f"{weighted_mean:.3f}", 
                    ha='center', va='bottom', fontsize=9)
            
            # For the first alpha, store size reference values for legend
            if j == 0:
                # Get min, median, and max counts for reference
                min_count = total_count.min()
                median_count = total_count.median()
                max_count = total_count.max()
                
                # Store values for legend
                size_reference_values = [min_count, median_count, max_count]
                size_reference_labels = [f"Min: {int(min_count)}", 
                                        f"Median: {int(median_count)}", 
                                        f"Max: {int(max_count)}"]
    
    # Set x-ticks at alpha positions
    plt.xticks(range(len(alphas)), [f'α={alpha}' for alpha in alphas])
    plt.xlabel('Method')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error by Method (circle size proportional to area size)')
    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Only create the size reference legend
    if size_reference_values:
        # Calculate sizes for legend
        legend_sizes = 100 * (np.array(size_reference_values) / size_reference_values[-1])
        
        # Create legend elements - using unfilled circles
        legend_elements = []
        for size, label in zip(legend_sizes, size_reference_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             label=label, markerfacecolor='none', 
                                             markeredgecolor='black', markersize=np.sqrt(size/np.pi)))
        
        # Add legend for size reference
        plt.legend(handles=legend_elements, title="Area Size (points)", 
                  loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/proportion_error_by_size.jpeg", format="jpeg", dpi=300)
    plt.show()
    plt.close()
    
    # Modify the existing boxplot to add a legend explaining the visualization
    plt.figure(figsize=(10, 8))
    
    abs_errors_by_alpha = []
    labels = []
    
    for alpha in alphas:
        alpha_str = str(float(alpha))
        pred_col = f'alpha_{alpha_str}_run_1_proportion'
        
        if pred_col in proportion_comparison_df.columns:
            errors = abs(proportion_comparison_df[pred_col] - proportion_comparison_df['target_proportion'])
            abs_errors_by_alpha.append(errors)
            alpha_idx = alphas.index(alpha)
            labels.append(PARAMETER_LABELS[alpha_idx])
    
    plt.boxplot(abs_errors_by_alpha, labels=labels)
    plt.ylabel('Absolute Error')
    plt.title('Distribution of Absolute Errors by Method')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text explaining the visualization
    plt.figtext(0.5, 0.01, 
                "Note: See 'proportion_error_by_size.jpeg' for visualization with circle sizes\n" +
                "proportional to area size (number of data points).",
                ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text
    plt.savefig(f"{OUTPUT_DIR}/proportion_error_boxplot.jpeg", format="jpeg", dpi=300)
    plt.show()
    plt.close()


def plot_count_visualizations(proportion_comparison_df: pd.DataFrame, alphas: List[float] = ALPHAS):
    """
    Create visualizations that incorporate count information for better understanding
    of the predictions across areas.
    
    Args:
        proportion_comparison_df: DataFrame with area-level proportions and counts
    """
    # Create a figure with subplots for each alpha
    fig = plt.figure(figsize=(22, 8))  # Wider figure to accommodate colorbar
    
    # Create a GridSpec layout with space for the colorbar
    gs = fig.add_gridspec(1, len(alphas) + 1, width_ratios=[1] * len(alphas) + [0.1])
    
    axes = []
    for i in range(len(alphas)):
        axes.append(fig.add_subplot(gs[0, i]))
    
    # Calculate the correct total counts for each area
    # Use target counts which should be consistent across all alphas
    proportion_comparison_df['total_count'] = proportion_comparison_df['target_count_0'] + proportion_comparison_df['target_count_1']
    
    # Use multiple alpha values for comparison
    for i, alpha in enumerate(alphas):
        ax = axes[i]
        
        alpha_str = str(float(alpha))
        pred_col = f'alpha_{alpha_str}_run_1_proportion'
        
        if pred_col in proportion_comparison_df.columns:
            # Calculate absolute error
            proportion_comparison_df['abs_error'] = abs(
                proportion_comparison_df[pred_col] - proportion_comparison_df['target_proportion']
            )
            
            # Create bins for area size and target proportion
            # Use logarithmic scale for area size to better represent the distribution
            if proportion_comparison_df['total_count'].min() > 0:
                area_bins = np.logspace(
                    np.log10(proportion_comparison_df['total_count'].min()),
                    np.log10(proportion_comparison_df['total_count'].max()),
                    10
                )
            else:
                # Handle case where min value is 0
                min_nonzero = proportion_comparison_df['total_count'][proportion_comparison_df['total_count'] > 0].min()
                area_bins = np.logspace(
                    np.log10(min_nonzero),
                    np.log10(proportion_comparison_df['total_count'].max()),
                    10
                )
            
            prop_bins = np.linspace(0, 1, 10)
            
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(
                proportion_comparison_df['total_count'],
                proportion_comparison_df['target_proportion'],
                bins=[area_bins, prop_bins],
                weights=proportion_comparison_df['abs_error']
            )
            
            # Count number of points in each bin for averaging
            counts, _, _ = np.histogram2d(
                proportion_comparison_df['total_count'],
                proportion_comparison_df['target_proportion'],
                bins=[area_bins, prop_bins]
            )
            
            # Avoid division by zero
            counts[counts == 0] = 1
            
            # Calculate average error in each bin
            avg_error = hist / counts
            
            # Create heatmap
            im = ax.imshow(
                avg_error.T,
                origin='lower',
                aspect='auto',
                extent=[np.log10(area_bins[0]), np.log10(area_bins[-1]), yedges[0], yedges[-1]],
                cmap='YlOrRd'
            )
            
            # Set logarithmic x-axis with appropriate labels
            ax.set_xlabel('Area Size (log scale)')
            # Create nicer x-tick labels at powers of 10
            log_ticks = np.arange(np.ceil(np.log10(area_bins[0])), np.floor(np.log10(area_bins[-1]))+1)
            ax.set_xticks(log_ticks)
            ax.set_xticklabels([f'10^{int(x)}' for x in log_ticks])
            
            # Only add y-label to the leftmost column
            if i == 0:
                ax.set_ylabel('Target Proportion')
            
            # Add alpha value as title for each subplot
            ax.set_title(f'α = {alpha}')
    
    # Add a colorbar in the dedicated space
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Average Absolute Error')
    
    # Add a single title for the entire figure
    fig.suptitle('Error by Area Size and Target Proportion', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/error_heatmap_by_size_and_proportion.jpeg", format="jpeg", dpi=300)
    plt.show()
    plt.close()


# %%
# Load all predictions into a hierarchical structure for EDA
print("Loading predictions data...")
predictions_by_params = load_predictions()



# %%

# If needed: Combine all predictions into a single DataFrame
# all_predictions_df = combine_predictions(predictions_by_params)
# all_predictions_df.to_csv(f"{OUTPUT_DIR}/all_predictions.csv", index=False)

# Calculate metrics and create DataFrame for analysis
print("Calculating metrics...")
results_df = calculate_metrics(predictions_by_params)

# Save the metrics DataFrame for further analysis
# results_df.to_csv(f"{OUTPUT_DIR}/experiment_results.csv", index=False)
# print(f"Metrics saved to {OUTPUT_DIR}/experiment_results.csv")

# After calculating metrics and before generating plots
print("Creating proportion comparison DataFrame...")
proportion_comparison_df = create_proportion_comparison_df(predictions_by_params)
proportion_comparison_df.to_csv(f"{OUTPUT_DIR}/proportion_comparison.csv", index=False)
print(f"Proportion comparison saved to {OUTPUT_DIR}/proportion_comparison.csv")

# %%
# Plot metrics
print("Generating plots...")
plot_metrics(results_df, 'classification', list(CLASSIFICATION_METRICS.keys()))
plot_metrics(results_df, 'proportion', list(PROPORTION_METRICS.keys()))
create_summary_plot(results_df)
create_pareto_plot(results_df)

# %%
print("Generating proportion comparison plots...")
plot_proportion_comparisons(proportion_comparison_df)

# %%
print("Generating count visualizations...")
plot_count_visualizations(proportion_comparison_df)

print(f"All plots saved to {OUTPUT_DIR}")
# %%
