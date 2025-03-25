#%%
from sdm.run import run, HNNArgs

hnn_experiments = [
    [
        HNNArgs(
            run_name=f"hnn_alpha-{alpha}_train-{train_perc/100}",
            points_data_path="data/lancaster/processed/points_data_oa_with_area_num.csv",
            output_dir="results/lancaster/test",
            use_seperate_proportions = False,
            alpha=alpha,
            n_iter=200,
            n_runs=10,
            train_prop=train_perc/100,
            filter_to_aoi=False,
        ) for train_perc in range(0, 40, 10) # Can not have 100% training data since some data is required to validate the model.
    ] for alpha in [0.1, 0.5]
]

for alpha in hnn_experiments:
    for train_perc in alpha:
        run(train_perc)
# %%
# random_experiments = [
# HNNArgs(
#         run_name=f"hnn_alpha-{0}_train-{train_perc/100}",
#         points_data_path="data/lancaster/processed/points_data_oa_with_area_num.csv",
#         output_dir="results/lancaster/test",
#         use_seperate_proportions = False,
#         alpha=0,
#         n_iter=1, #random disaggregation only needs 1 iteration
#         n_runs=10,
#         train_prop=train_perc/100,
#         filter_to_aoi=False,
#     ) for train_perc in range(0, 40, 10)] # Can not have 100% training data since some data is required to validate the model.

# for e in random_experiments:
#     run(e)