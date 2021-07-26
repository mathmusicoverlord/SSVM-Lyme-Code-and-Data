"""
Script 1: Script for used to remove features between healthy control groups. Our results are stored in the .csv files:
"../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_knn_[normalization]_removed_features_healthy_controls.csv",
where [normalization] is either log, standard, medianfoldchange, or none. This script should
be run in segments, as there are hyper-parameter tuning steps which require variable user input.
"""


if __name__ == "__main__":
    # imports
    import Metabolomics
    import datetime
    import os

    # Set parameters
    classifier = 'ssvm'
    normalization_method = 'log'
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

    # Remove Lidocaine from analysis
    x.remove_feature(identifier=[487, 489])

    # Normalize and Impute
    x.normalize(algorithm=normalization_method)
    x.impute(algorithm=imputation_method)

    # Visualize pre-IFR
    x.visualize(attrname='Disease State', dimension=2, method='umap', save=True, save_ext='pre_ifr')

    # Remove features that split healthy groups
    healthy = x.partition('Health State')['healthy']
    x.iter_feature_removal(attrname='Disease State',
                           idx_list=healthy,
                           C=1,
                           cv=2,
                           save_all_classifiers=True,
                           num_features=5,
                           threshold=.6,
                           method='ssvm',
                           save=True)

    # visualize data post ifr
    x.visualize(attrname='Disease State', dimension=2, method=viz_method, save=True, save_ext='post_ifr')