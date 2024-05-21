# Code for [paper name hidden for double blind peer review]

To install the package and dependencies, run:
``` 
pip install -e .
pip install -r requirements.txt
```

Run experiments/hnn_val.py for HNN validation. The results will be stored in the results directory. Use scripts/plotting.py with FLOODMAP_EVALS = False to plot the results. 

A floodmap shapefile is needed to run the flood damage estimation. Flood map zone 3 can be downloaded from https://www.data.gov.uk/dataset/bed63fc1-dd26-4685-b143-2941088923b3/flood-map-for-planning-rivers-and-sea-flood-zone-3. Store your shapefile in data/lancaster/floodmapR&S_zone3_lancaster/floodmap.shp.
Then run experiments/floodmap_damage.py for flood damage estimation. The results will be stored in the results/floodmap directory. Use scripts/plotting.py with FLOODMAP_EVALS = True to plot the results. 
