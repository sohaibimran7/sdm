#%%
from sdm.run import run, HNNArgs

random_experiments = [
    HNNArgs(
    run_name=f"floodmap/alpha-{0}_train-{train_perc/100}",
    points_data_path="data/lancaster/processed/points_data.csv",
    proportions_data_path="data/lancaster/processed/proportions_data.csv",
    alpha=0, 
    n_iter=1, #random disaggregation only needs 1 iteration
    n_runs=100,
    train_prop=train_perc/100,
    filter_to_aoi=True
    ) for train_perc in range(0, 101, 5)
]

for e in random_experiments:
    run(e)

# %%
