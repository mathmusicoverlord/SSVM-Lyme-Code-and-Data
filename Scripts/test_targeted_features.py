"""
Script 4: Script used to evaluate the test performance of the model built on training data targeted
at the selected features in Skyline. Our test scores are stored to the open doc spreadsheet
"../Data/20210211_ELvsHCEric_BF_BestMFsTransitionResults_2_test_scores.ods". This script should be run in segments,
as there are hyper-parameter tuning steps which require variable user input.
"""

if __name__ == "__main__":
    # imports
    import Metabolomics
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import datetime
    import os

    # Set parameters
    classifier = 'ssvm'
    viz_method = 'mds'
    normalization_method = 'log'

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
        data_file_path='../Data/20210211_ELvsHCEric_BF_BestMFsTransitionResults_2.csv',
        metadata_file_path='../Data/HCvsEL_clinicaldata_102519_gen.csv', data_format="Skyline",
        metadata_format=0, osmolality_file_path="")

    # set save path
    x.path = save_loc

    # normalize
    x.normalize(algorithm=normalization_method)

    # visualize generations
    x.visualize(attrname='Generation', method='mds', save=True)

    # visualize lyme vs. healthy
    x.visualize(attrname='Health State', method='mds', save=True)

    # visualize by disease state
    x.visualize(attrname='Disease State', method='mds', save=True)

    # Identify old serum for training/validation
    training_id = x.partition(attrname='Generation')['old']
    test_id = x.partition(attrname='Generation')['new']
    n_training = len(training_id)
    n_test = len(test_id)

    # tune C parameter for ssvm
    x.classify(method=classifier, attrname='Health State', idx_list=training_id, cv=5, C=np.arange(1, 10, .2), viz=True, save=True)

    # train model
    experiment = x.classify(method=classifier, attrname='Health State', idx_list=training_id, cv=n_training, C=7.5)
    ssvm_classifier = experiment.best_classifiers['SSVMClassifier']

    # test model
    test_data = x.new_data[test_id, :]
    predict_labels = np.array(ssvm_classifier.predict(test_data))
    true_labels = x.generate_labels(attrname='Health State', idx_list=test_id)
    disease_state_labels = x.generate_labels('Disease State', idx_list=test_id)

    # scores
    C = confusion_matrix(true_labels, predict_labels, labels=['healthy', 'lyme'])
    acc = (C[0, 0] + C[1, 1]) / np.sum(C)  # accuracy score
    bsr = .5 * ((C[0, 0] / (C[0, 0] + C[0, 1])) + (C[1, 1] / (C[1, 1] + C[1, 0])))  # balanced success rate score

    # format scores
    C = pd.DataFrame(index=['True Healthy', 'True Lyme'], columns=['Predicted Healthy', 'Predicted Lyme'], data=C)
    C.to_csv(save_loc + '20210211_ELvsHCEric_BF_BestMFsTransitionResults_2_confusion_mat.csv')

    # split by disease state
    C_by_disease_state = pd.DataFrame(index=['HCW', 'HCHu', 'EDL', 'ELL'], columns=['Predicted Healthy', 'Predicted Lyme'])
    edl = disease_state_labels == 'edl'
    edl_bins = confusion_matrix(true_labels[edl], predict_labels[edl], labels=['healthy', 'lyme'])[1,:]
    ell = disease_state_labels == 'ell'
    ell_bins = confusion_matrix(true_labels[ell], predict_labels[ell], labels=['healthy', 'lyme'])[1,:]
    hcw = disease_state_labels == 'hcw'
    hcw_bins = confusion_matrix(true_labels[hcw], predict_labels[hcw], labels=['healthy', 'lyme'])[0,:]
    hchu = disease_state_labels == 'hchu'
    hchu_bins = confusion_matrix(true_labels[hchu], predict_labels[hchu], labels=['healthy', 'lyme'])[0,:]
    C_by_disease_state.loc['HCW'] = hcw_bins
    C_by_disease_state.loc['HCHu'] = hchu_bins
    C_by_disease_state.loc['EDL'] = edl_bins
    C_by_disease_state.loc['ELL'] = ell_bins
    C_by_disease_state.to_csv(save_loc + '20210211_ELvsHCEric_BF_BestMFsTransitionResults_2_confusion_mat_by_disease_state.csv')

