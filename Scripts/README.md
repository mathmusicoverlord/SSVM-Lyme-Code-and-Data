This directory contains the python scripts used to generate the results for the manuscript: *Biomarker Selection and a Prospective Metabolite-based Machine Learning Diagnostic for Lyme Disease*

## Key

+ calcom directory
	- Contains the calcom python library used to perform calculations and build models. Read the README in the calcom directory
	  for instructions on how to install the library.

+ Metabolomics.py
	- Extension library to calcom for performing calculations relative to metabolomics. Contains quality of life features
	  for plotting and performing classification/feature selection experiments. calcom must be installed to use this library.

+ iterative_feature_removal.py
	- Script 1: Script for used to remove features between healthy control groups. Our results are stored in the .csv file: 
	  "../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_knn_log_removed_features_healthy_controls.csv". This script should
	  be run in segments, as there are hyper-parameter tuning steps which require variable user input.

+ feature_selection.py
	- Script 2: Script for used to perform feature selection post-ifr. Our results are store in the excel spreadsheet:
	  "../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_Features.xlsx". This script should be run in segments, as there
	  are hyper-parameter tuning steps which require variable user input.

+ combine_feature_sets.py
	- Script 3: Script used to combine feature sets across different imputation/normalization schemes. These features can be
	  sent for evaluation, peak quality, and targeting in Skyline. Our results are store in the excel spreadsheet:
	  "../Data/PeakTable_022315ELvsHC_afterfillpeaks_111819_Features.xlsx" in the sheet "ssvm_selected_features".

+ test_targeted_features.py
	- Script 4: Script used to evaluate the test performance of the model built on training data targeted
	  at the selected features in Skyline. Our test scores are stored to the open doc spreadsheet
	  "../Data/20210211_ELvsHCEric_BF_BestMFsTransitionResults_2_test_scores.ods". This script should be run 
	  in segments, as there are hyper-parameter tuning steps which require variable user input.

+ test_targeted_log_features.py
	- Script 4 (opt): Script 4 (opt): Script used to evaluate the test performance of the model built on training data targeted
	  at the selected log features in Skyline. Our test scores are stored to the open doc spreadsheet
	  "../Data/20210211_ELvsHCEric_BF_BestMFsTransitionResults_2_test_scores.ods". This script should be run in segments,
	  as there are hyper-parameter tuning steps which require variable user input.

