#%%
from sdm.run import run, HNNArgs

hnn_experiments = [
    HNNArgs(
    run_name=f"floodmap/alpha-{0.1}_train-{train_perc/100}",
    points_data_path="data/lancaster/processed/points_data.csv",
    proportions_data_path="data/lancaster/processed/proportions_data.csv",
    alpha=0.1, 
    n_iter=200,
    n_runs = 100,
    train_prop=train_perc/100,
    filter_to_aoi=True
    ) for train_perc in range(0, 101, 5)
]

for e in hnn_experiments:
    run(e)

# %%
