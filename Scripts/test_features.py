# load data
# imports
import Metabolomics
import datetime
import os
import pandas as pd
import numpy as np

# Set parameters
classifier = 'ssvm'
normalization_method = 'standard'
imputation_method = 'knn'
viz_method = 'umap'

# Set date and save location
dt = datetime.datetime.now()
d_truncated = datetime.date(dt.year, dt.month, dt.day)
save_loc = '../Data/Results/' + d_truncated.__str__()

# Create new directory
try:
    os.mkdir(save_loc)
except Exception:
    pass
save_loc = save_loc + '/'

# Load in dataset
x = Metabolomics.CCLymeMetaboliteSet(
    data_file_path='../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819.csv',
    metadata_file_path='../Data/HCvsEL_clinicaldata_102519_gen.csv',
    data_format="XCMS",
    metadata_format=0, osmolality_file_path="")

# set save path
x.path = save_loc

# Normalize and Impute
x.normalize(algorithm=normalization_method, controls=np.arange(118))
x.impute(algorithm=imputation_method)

# set features
all_features = pd.read_excel(open('../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_Features.xlsx', 'rb'),
                         sheet_name='ssvm',
                         engine='openpyxl',
                         header=3)

median_features = all_features.filter(regex='id_\d$')
log_features = all_features.filter(regex='id_\d.1$')
none_features = all_features.filter(regex='id_\d.2$')
standard_features = all_features.filter(regex='id_\d.3$')
feature_set = none_features

# check we used the same feature list
feature_sets = []
for column in feature_set:
    features = feature_set[column].unique()
    features.sort()
    features = features[:-1].astype(int)
    feature_sets.append(features)

# check if they are all the same
tf = (feature_sets[0] == feature_sets[1]).all() and (feature_sets[0] == feature_sets[2]).all() and (feature_sets[0] == feature_sets[3]).all() and (feature_sets[0] == feature_sets[4]).all()
print(tf)

if tf:
    feature_set = feature_sets[0].tolist()
    ifr_features = list(set(np.arange(4851).tolist()).difference(set(feature_set)))

# save features
df_ifr_features = x.varattr[x.varattr['id'].isin(ifr_features)]
df_ifr_features.to_csv('../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_knn_none_removed_features_healthy_controls.csv')

# visualize disease state
x.visualize(attrname='Disease State', method='umap', marker_size=200, save=True, save_ext='pre_ifr')
x.visualize(attrname='Disease State', features=feature_set, method='umap', marker_size=200, save=True, save_ext='post_ifr')
x.visualize(attrname='Disease State', features=ifr_features, method='umap', marker_size=200, save=True, save_ext='just_ifr')