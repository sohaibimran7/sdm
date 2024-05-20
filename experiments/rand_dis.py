#%%
from sdm.run import run, HNNArgs

random_experiments = [
HNNArgs(
        run_name=f"alpha-{0}_train-{train_perc/100}",
        points_data_path="data/lancaster/processed/points_data.csv",
        use_seperate_proportions = False,
        alpha=0,
        n_iter=1, #random disaggregation only needs 1 iteration
        n_runs=100,
        train_prop=train_perc/100,
        filter_to_aoi=False,
    ) for train_perc in range(0, 96, 5)]

for e in random_experiments:
    run(e)