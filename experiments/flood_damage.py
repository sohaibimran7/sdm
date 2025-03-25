# %%
from sdm.run import run, HNNArgs

hnn_experiments = [
    HNNArgs(
        run_name=f"floodmap/hnn_alpha-{0.1}_train-{train_perc/100}",
        points_data_path="data/lancaster/processed/points_data.csv",
        proportions_data_path="data/lancaster/processed/proportions_data.csv",
        alpha=0.1,
        n_iter=200,
        n_runs=100,
        train_prop=train_perc / 100,
        filter_to_aoi=True,
        # Flood map zone 3 can be downloaded from https://www.data.gov.uk/dataset/bed63fc1-dd26-4685-b143-2941088923b3/flood-map-for-planning-rivers-and-sea-flood-zone-3
        aoi_path="data/lancaster/floodmapR&S_zone3_lancaster/floodmap.shp",
    )
    for train_perc in range(0, 101, 5)
]

for e in hnn_experiments:
    run(e)

# %%
random_experiments = [
    HNNArgs(
        run_name=f"floodmap/hnn_alpha-{0}_train-{train_perc/100}",
        points_data_path="data/lancaster/processed/points_data.csv",
        proportions_data_path="data/lancaster/processed/proportions_data.csv",
        alpha=0,
        n_iter=1,  # random disaggregation only needs 1 iteration
        n_runs=100,
        train_prop=train_perc / 100,
        filter_to_aoi=True,
        # Flood map zone 3 can be downloaded from https://www.data.gov.uk/dataset/bed63fc1-dd26-4685-b143-2941088923b3/flood-map-for-planning-rivers-and-sea-flood-zone-3
        aoi_path="data/lancaster/floodmapR&S_zone3_lancaster/floodmap.shp",
    )
    for train_perc in range(0, 101, 5)
]

for e in random_experiments:
    run(e)

# %%
from sdm.run import run, HNNArgs, KNNArgs

knn_experiments = [
    [
        KNNArgs(
            run_name=f"floodmap/k-{k}_train-{train_perc/100}",
            points_data_path="data/lancaster/processed/points_data.csv",
            proportions_data_path="data/lancaster/processed/proportions_data.csv",
            k=k,
            n_runs=100,
            train_prop=train_perc / 100,
            filter_to_aoi=True,
            # Flood map zone 3 can be downloaded from https://www.data.gov.uk/dataset/bed63fc1-dd26-4685-b143-2941088923b3/flood-map-for-planning-rivers-and-sea-flood-zone-3
            aoi_path="data/lancaster/floodmapR&S_zone3_lancaster/floodmap.shp",
        )
        for train_perc in range(5, 101, 5)
    ]
    for k in (1, 3)
]

for k in knn_experiments:
    for train_perc in k:
        run(train_perc)

# %%
