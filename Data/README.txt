This directory contains the data for the 
manuscript: Biomarker Selection and Classification of LC-MS SerumLyme with SSVM
  ___________
 //// Key \\\\
****	   ****
+ PeakTable_022315ELvsHC_afterfillpeaks_111819.csv:
	- XCMS features for the training (old) serum samples
	- 118 samples:
		* 30 Early Disseminated Lyme (EDL)
		* 30 Early Localized Lyme (ELL)
		* 28 Healthy Control Non-Endemic (HCN)
		* 30 Healthy Control Wormser (HCW)
	- 4,851 dimensional feature vector of intensities for (mass/charge, retention time) measured
	  by an Agilent mass spectrometer and processed by the R-package XCMS.

+ 20210211_ELvsHCEric_BF_BestMFsTransitionResults_2.csv:
	- Targeted Skyline features for both training (old) and test (new) samples
	- 118 samples:
		* 40 Early Disseminated Lyme (EDL)​
		* 40 Early Localized Lyme (ELL)​
		* 30 Healthy Control Wormser (HCW)​
		* 8 Healthy Hu (HCHU)
	- 42-dimensional feature vectors of intensities processed by Skyline.
	  Training and Test serum data sets targeted in Skyline at the select features.

+ HCvsEL_clinicaldata_102519_gen.csv:
	- Metadata/clinical data for training and test samples
	- Generation = old --> training
	- Generation = new --> test

+ PeakTable_022315ELvsHC_afterfillpeaks_111819_knn_[method]_removed_features_healthy_controls.csv:
	- Features removed by iterative feature removal to bring healthy control groups
	  HCN and HCW together in the training data. See the script iterative_feature_removal.py.
	- [method] can either be log, standard, medianfoldchange, or none depending on the normalization used.

+ PeakTable_022315ELvsHC_afterfillpeaks_111819_Features.xlsx:
	- ssvm Sheet:
		* Contains the XCMS feature sets chosen by kFFS and SSVM with k=5 and n=5 across imputation 
		  and normalization schemes: knn/log, knn/standard/, knn/median-fold change, and knn/none.
		  The features highlighted in yellow are the top features for their fold and orange
		  indicates the top 5 from each fold. The green columns indicate the cumulative top 5
		  accross folds for each scheme.

	- ssvm_selected_features Sheet:
		* Combines the kFFS features across imputation and normalization schemes. These were the features
		  which were sent for targeting in Skyline.

+ 20210211_ELvsHCEric_BF_BestMFsTransitionResults_2_test_scores.ods
	- Prediction results of SSVM trained on training data and tested on test data.

+ 20210322_PeakTable_022315ELvsHC_afterfillpeaks_111819_feature_set_msms
	- Contains MSMS information for Skyline targeted features.

+ Results directory
	- Contains the output of scripts run by the user. See "../Scripts/"