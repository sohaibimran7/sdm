#%%
from sdm.run import run, HNNArgs

hnn_experiments = [
    HNNArgs(
        run_name=f"hnn_alpha-{0.1}_train-{0/100}_dt-{dt}",
        points_data_path="data/lancaster/processed/points_data.csv",
        output_dir="results/lancaster/dt_test",
        use_seperate_proportions = False,
        alpha=0.1,
        n_iter=10000,
        n_runs=1,
        train_prop=0/100,
        filter_to_aoi=False,
        dt=dt,
    ) for dt in [0.01, 0.1, 1.0] # Can not have 100% training data since some data is required to validate the model.
]


for e in hnn_experiments:
    run(e)
# %%
# random_experiments = [
# HNNArgs(
#         run_name=f"hnn_alpha-{0}_train-{train_perc/100}",
#         points_data_path="data/lancaster/processed/points_data.csv",
#         use_seperate_proportions = False,
#         alpha=0,
#         n_iter=1, #random disaggregation only needs 1 iteration
#         n_runs=3,
#         train_prop=train_perc/100,
#         filter_to_aoi=False,
#         output_dir="results/lancaster/oa",
#     ) for train_perc in [0, 20, 50]] # Can not have 100% training data since some data is required to validate the model.

# for e in random_experiments:
#     run(e)