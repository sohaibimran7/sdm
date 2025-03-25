#%%
from sdm.run import run, KNNArgs

# Test different k values and training proportions
k_values = [1, 3]
train_percentages = range(5, 96, 40)  # 5% to 95% in steps of 5%

knn_experiments = [
    [
        KNNArgs(
            run_name=f"k-{k}_train-{train_perc/100}",
            points_data_path="data/lancaster/processed/points_data.csv",
            use_seperate_proportions=False,
            k=k,
            n_runs=2,
            train_prop=train_perc/100,
            filter_to_aoi=False,
        ) for train_perc in train_percentages
    ] for k in k_values
]

# Run experiments
for k_configs in knn_experiments:
    for config in k_configs:
        run(config) 