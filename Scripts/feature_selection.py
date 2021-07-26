"""
Script 2: Script for used to perform feature selection post-ifr. Our results are store in the excel spreadsheet:
"../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_Features.xlsx". This script should be run in segments, as there
are hyper-parameter tuning steps which require variable user input.
"""


if __name__ == "__main__":
    # imports
    import Metabolomics
    import numpy as np
    import pandas as pd
    import datetime
    import os

    # Set parameters
    classifier = 'ssvm'
    viz_method = 'mds'
    normalization_method = 'log'  # other options 'standard', 'None', 'median-fold change'
    imputation_method = 'knn'

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

    # remove features found by ifr
    removed_features = pd.read_csv(
        '../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_knn_log_removed_features_healthy_controls.csv') # our removed features
    #removed_features = pd.read_csv(
    #    save_loc + 'PeakTable_022315ELvsHC_afterfillpeaks_111819_knn_log_removed_features_healthy_controls.csv') # your removed features
    x.remove_feature(removed_features['id'].values)

    # normalize and impute
    x.normalize(algorithm=normalization_method)
    x.impute(algorithm=imputation_method)

    # Run classifier and save
    x.classify(method=classifier, attrname='Health State', cv=5, C=np.arange(0, 3, .03), viz=True, save=True) # tuning step

    # Choose C parameter, feature select, and save
    [df, num_top_features, acc] = x.feature_select(method=classifier, attrname='Health State', C=.95, save=True, save_all_classifiers=True)

    # Grab Top 5 Features from each and collaspe
    feature_list = np.array([])  # Initialize
    for index in range(5):
        feature_list = np.append(feature_list, df['order_' + str(index)].to_list()[0:5])

    # Count occurences
    feature_list = [int(item) for item in feature_list]
    occurences = [[j, feature_list.count(j)] for j in set(feature_list)]
    feature_list = np.unique([int(item) for item in feature_list])

    # Grab Cumulative Top 5 Features
    df_top = x.varattr.loc[feature_list, ['id', 'rt', 'mz', 'Proportion Missing']]

    # Add occurence column
    df_top.loc[:, 'Occurence'] = 0  # Initialize
    for pair in occurences:
        df_top.at[pair[0], 'Occurence'] = pair[1]

    # Sort Cumulative Top 5
    df_top.sort_values(by=['Occurence'], inplace=True, ascending=False)

    # Save Cumulative Top 5
    df_top.to_csv(path_or_buf=x.path + x.name + '_' + classifier + '_Health_State_' + imputation_method + '_' + normalization_method + '_top_5_features.csv')

    # Get accuracy off Top 5
    x.classify(method=classifier, feature_set=feature_list, cv=5, C=[1])

    # Visualize post-selection and save
    x.visualize(features=feature_list, attrname='Disease State', dimension=2, method=viz_method, save=True, save_ext='post_feature_selection')