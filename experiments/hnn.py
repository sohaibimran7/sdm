#%%
from sdm.run import run, HNNArgs

hnn_experiments = [
    [
        HNNArgs(
            run_name=f"alpha-{alpha}_train-{train_perc/100}",
            points_data_path="data/lancaster/processed/points_data.csv",
            use_seperate_proportions = False,
            alpha=alpha,
            n_iter=200,
            n_runs=100,
            train_prop=train_perc/100,
            filter_to_aoi=False,
        ) for train_perc in range(0, 96, 5)
    ] for alpha in [0.1, 0.5, 0.9]
]

for alpha in hnn_experiments:
    for train_perc in alpha:
        run(train_perc)
# %%
