#%%
from sdm.run import run, HNNArgs

# This file executes 2 runs (instead of the usual 100) of all experiments with 
# 0, 40 and 80% properties seeded instead of every 5% interval 
# Results are saved to results/examples directory 
# This allows testing the code before running the experiments
#%%
random_experiments = [
HNNArgs(
        run_name=f"example/alpha-{0}_train-{train_perc/100}",
        points_data_path="data/lancaster/processed/points_data.csv",
        use_seperate_proportions = False,
        alpha=0,
        n_iter=1, #random disaggregation only needs 1 iteration
        n_runs=2,
        train_prop=train_perc/100,
        filter_to_aoi=False,
    ) for train_perc in range(0, 96, 40)]

for e in random_experiments:
    run(e)
#%%
hnn_experiments = [
    [
        HNNArgs(
            run_name=f"example/alpha-{alpha}_train-{train_perc/100}",
            points_data_path="data/lancaster/processed/points_data.csv",
            use_seperate_proportions = False,
            alpha=alpha,
            n_iter=200,
            n_runs=2,
            train_prop=train_perc/100,
            filter_to_aoi=False,
        ) for train_perc in range(0, 96, 40)
    ] for alpha in [0.1, 0.5, 0.9]
]

for alpha in hnn_experiments:
    for train_perc in alpha:
        run(train_perc)
#%%
random_experiments = [
    HNNArgs(
    run_name=f"example/floodmap/alpha-{0}_train-{train_perc/100}",
    points_data_path="data/lancaster/processed/points_data.csv",
    proportions_data_path="data/lancaster/processed/proportions_data.csv",
    alpha=0, 
    n_iter=1, #random disaggregation only needs 1 iteration
    n_runs=2,
    train_prop=train_perc/100,
    filter_to_aoi=True
    ) for train_perc in range(0, 101, 40)
]

for e in random_experiments:
    run(e)
#%%
hnn_experiments = [
    HNNArgs(
    run_name=f"example/floodmap/alpha-{0.1}_train-{train_perc/100}",
    points_data_path="data/lancaster/processed/points_data.csv",
    proportions_data_path="data/lancaster/processed/proportions_data.csv",
    alpha=0.1, 
    n_iter=200,
    n_runs = 2,
    train_prop=train_perc/100,
    filter_to_aoi=True
    ) for train_perc in range(0, 101, 40)
]

for e in hnn_experiments:
    run(e)