# %%
import pandas as pd
from matplotlib import pyplot as plt

# %%
LABELS_PATH = "data/lancaster/processed/uprn_labels.csv"
COUNTS_PATH = "data/lancaster/processed/uprn_counts.csv"

# %%
labels = pd.read_csv(LABELS_PATH)
counts = pd.read_csv(COUNTS_PATH)
# %%
tap_sortkey = counts[['areas', 'nonres_proportion']].sort_values(by='nonres_proportion', ascending=False)['areas']
tap_sortkey

# %%
grouped_labels = labels.dropna(subset=['label']).groupby('areas')['label']
nonres_validation = grouped_labels.sum()/grouped_labels.count()

lsoa_props_validation = labels.dropna(subset=['label']).groupby('areas').count()['uprn']/labels.groupby('areas').count()['uprn']

# %%

width = 0.2

plt.figure(figsize=(12,4))

plt.bar(counts['areas']-width/2, counts['nonres_proportion'].sort_values(ascending=False), width, label='target non-residential proportions')
plt.bar(counts['areas']+width/2, nonres_validation[tap_sortkey], width, label='non-residential proportions in validation data')


plt.legend(loc='upper right')

plt.xlabel('Area')
plt.ylabel('Non-residential proportion') 

plt.xlim([0, 90])
plt.show()

# %%
width = 0.2

plt.figure(figsize=(12,4))

plt.bar(counts['areas']-width/2, counts['nonres_proportion'].sort_values(ascending=False), width, label='target non-residential proportions')
plt.bar(counts['areas']+width/2, lsoa_props_validation[tap_sortkey], width, label='proportion of properties present in the validation data')


plt.legend(loc='upper right', prop={'size': 8})

plt.xlabel('Area')
plt.ylabel('Non-residential proportion') 

plt.xlim([0, 90])
plt.show()