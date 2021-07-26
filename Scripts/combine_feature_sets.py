"""
Script 3: Script used to combine feature sets across different imputation/normalization schemes. These features can be
sent for evaluation, peak quality, and targeting in Skyline. Our results are store in the excel spreadsheet:
"../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_Features.xlsx" in the sheet "ssvm_selected_features".
"""

if __name__ == "__main__":
    # imports
    import Metabolomics
    import numpy as np
    import pandas as pd
    import datetime
    import os

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

    # collected top feature sets
    feature_set_knn_log = pd.read_csv(
        save_loc + 'PeakTable_022315ELvsHC_afterfillpeaks_111819_ssvm_Health_State_knn_log_top_5_features.csv')
    feature_set_knn_standard = pd.read_csv(
        save_loc + 'PeakTable_022315ELvsHC_afterfillpeaks_111819_ssvm_Health_State_knn_standard_top_5_features.csv')
    feature_set_knn_median = pd.read_csv(
        save_loc + 'PeakTable_022315ELvsHC_afterfillpeaks_111819_ssvm_Health_State_knn_median-fold change_top_5_features.csv')
    feature_set_knn_none = pd.read_csv(
        save_loc + 'PeakTable_022315ELvsHC_afterfillpeaks_111819_ssvm_Health_State_knn_None_top_5_features.csv')

    # total features
    total_features = pd.concat((feature_set_knn_log, feature_set_knn_standard, feature_set_knn_median, feature_set_knn_none))
    total_features = total_features.groupby('id').agg(np.max).drop('Unnamed: 0', axis=1)
    total_features = total_features.iloc[(-total_features['Occurence']).argsort()]

    # save
    total_features.to_csv(save_loc + 'PeakTable_022315ELvsHC_afterfillpeaks_111819_ssvm_Health_State_selected_features.csv')