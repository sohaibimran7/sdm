#%% imports
import pandas as pd
import numpy as np

#%% dataset paths (Data already trimmed to the study area)
CENSUS_PATH = "data/lancaster/lancaster_dwellings_2011_LSOA.csv" 
UPRN_COUNTS_PATH = "data/lancaster/NSULlancasterUPRNcountsLSOA.csv"
UPRN_LOCS_PATH = "data/lancaster/UPRNlocslancasterLSOA.csv" 
EPC_PATH = "data/datasets/all-domestic-certificates/domestic-E07000121-Lancaster/certificates.csv"
OSPOI_PATH = "data/datasets/poi_GB_MAR21_csv/poi-extract-2021_03.csv"

# Total number of dwellings in Live tables on dwelling stock (including vacants) - Table 100: number of dwellings by tenure and district, England.
TOTAL_DWELLINGS_UPDATED = 64641 # hardcoded value, update for your analyses

OUTPUT_DIR = "data/lancaster/processed/"
OUTPUT_UPRN_LABELS_PATH = OUTPUT_DIR + "uprn_labels.csv"
OUTPUT_UPRN_COUNTS_PATH = OUTPUT_DIR + "uprn_counts.csv"

# %%
census_dwellings = pd.read_csv(CENSUS_PATH, skiprows=8, index_col=0, nrows=89).rename(lambda x: x.split(' : ')[0])
census_dwellings
# %%
census_dwellings["All categories: Dwelling type"].sum()

# %% [markdown]
# Since dwellings in 'Caravan or other mobile or temporary structure' are unlikely to have UPRNs, they need to be subtracted from total number of Dwellings to obtain Permanent dwellings. Since dwellings in 'Caravan or other mobile or temporary structure' are likely to be Unshared dwellings, we can directly subtract them from All categories: Dwelling type, to obtain 'Permanent Dwellings'

# %%
census_dwellings["Permanent_Dwellings"] = census_dwellings["All categories: Dwelling type"] - census_dwellings["Caravan or other mobile or temporary structure"]
census_dwellings

# %% [markdown]
# To get an estimate of permanent dwellings in Lancaster in 2021, we increase each Lancaster MSOA region's dwelling count by the same percentage as the entirity of Lanaster district's dwelling count increase, available via Live tables on dwelling stock (including vacants) - Table 100: number of dwellings by tenure and district, England.

# %%
dwellings = (census_dwellings["Permanent_Dwellings"] * TOTAL_DWELLINGS_UPDATED/census_dwellings["All categories: Dwelling type"].sum()).round()
dwellings

# %%
dwellings.sum()

# %% [markdown]
# ### UPRN counts data

# %%
uprn_counts = pd.read_csv(UPRN_COUNTS_PATH).rename(columns={"lsoa11cd":"LSOA11CD"})
uprn_counts

# %%
counts = uprn_counts.merge(dwellings, left_on="LSOA11CD", right_index=True)
counts

# %% [markdown]
# ### Calculating Proportions

# %%
counts['NonRes_UPRNs'] = counts['UPRNcount'] - counts['Permanent_Dwellings']
counts['NonRes_proportion'] = 1 - counts["Permanent_Dwellings"]/counts["UPRNcount"].to_numpy()
counts

# %%
np.sum(counts['NonRes_UPRNs'])/np.sum(counts['UPRNcount'])

# %%


# %%
counts.loc[counts['NonRes_proportion'] < 0, 'NonRes_proportion'] = 0

# %%
counts = counts.sort_values('LSOA11CD').reset_index(drop=True)
counts["Areas"] = np.arange(1, counts.shape[0] + 1)
counts

# %% [markdown]
# ## Point data
# 
# ### UPRNs to classify
# 
# UPRN point locations and LSOA regions data obtained by intersecting OSOPENUPRN data with LSOA

# %%
uprns = pd.read_csv(UPRN_LOCS_PATH, usecols=[1, 4, 5, 6])
uprns

# %%
uprns = uprns.merge(counts[['LSOA11CD', 'Areas']], on='LSOA11CD').sort_values("Areas")
uprns

# %% [markdown]
# ### Validation data

# %% [markdown]
# #### Residential - EPC

# %%
epc_properties = pd.read_csv(EPC_PATH)
epc_properties

# %%
epc_properties["INSPECTION_DATE"].sort_values()

# %%
epc_properties.dropna(subset=["UPRN"], inplace=True)
epc_properties.count()

# %%
np.unique(epc_properties["UPRN_SOURCE"], return_counts=True)

# %%
np.unique(np.unique(epc_properties["UPRN"], return_counts = True)[1], return_counts = True)

# %%
duplicates = epc_properties[epc_properties["UPRN"].duplicated(keep=False)]
duplicates = duplicates.sort_values("UPRN")
duplicates

# %%
epc_properties["label"] = 0
epc_properties

# %%
epc_uprns = epc_properties[["UPRN", "label"]].drop_duplicates(subset="UPRN", ignore_index=True)
epc_uprns

# %%
np.unique(np.unique(epc_uprns["UPRN"], return_counts = True)[1], return_counts = True)

# %% [markdown]
# #### Non residential - OS POI

# %%
ospoi_properties = pd.read_csv(OSPOI_PATH, delimiter="|").rename(columns={"uprn":"UPRN"}).merge(uprns, how="inner", on = "UPRN")
ospoi_properties

# %%
np.unique(np.unique(ospoi_properties["UPRN"], return_counts = True)[1], return_counts = True)

# %%
ospoi_properties["label"] = 1
ospoi_uprns = ospoi_properties[["UPRN", "label"]].drop_duplicates(subset="UPRN", ignore_index=True)
ospoi_uprns

# %% [markdown]
# ##### Combine datasets

# %%
labelled_uprns = pd.concat([ospoi_uprns, epc_uprns]).drop_duplicates(subset="UPRN", keep=False)
labelled_uprns

# %%
np.unique(labelled_uprns["label"], return_counts=True)

# %%
# Do not run this cell more than once. 
labelled_uprns = uprns.merge(labelled_uprns, how="left", on="UPRN")
labelled_uprns

# %%
labelled_uprns.rename(columns={
    "UPRN":"uprn", "LATITUDE": "latitude", "LONGITUDE" : "longitude", "LSOA11CD": "lsoa11cd", "Areas": "areas"}, inplace=True)

labelled_uprns.to_csv(OUTPUT_UPRN_LABELS_PATH)

#%%
counts.rename(columns={
    "UPRNcount":"uprn_count", 
    "LSOA11CD": "lsoa11cd", 
    "Areas": "areas", 
    "Permanent_Dwellings" : "res_uprns", 
    "NonRes_UPRNs":"nonres_uprns", 
    "NonRes_proportion":"nonres_proportion"}, inplace=True)

counts.to_csv(OUTPUT_UPRN_COUNTS_PATH)
# %%
