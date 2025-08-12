# Improved-AD-diagnosis-using-multimodal-sparse-similarity-feature-selection-and-auxiliary-data
The code of Improved Alzheimer’s disease diagnosis using multimodal sparse similarity feature selection and auxiliary data
The code consists of MATLAB script files.

1.Load the data into MATLAB.

2.Open the gridSearch.m file, which implements the multi-kernel SVM method. The function parameters include the kernel matrices of the MRI and PET training sets, the kernel matrices of the test sets, the sizes of the training and test sets, as well as the labels of the training and test sets.
The return values include accuracy, SEN (sensitivity), SPE (specificity), AUC, F1-score, and ROC coordinate values.

3.Run the Ours.m file, which implements the method described in the paper "Improved Alzheimer's Disease Diagnosis Using Multimodal Sparse Similarity Feature Selection and Auxiliary Data."

3.1 The code first balances the number of samples in both classes using undersampling.

3.2 It performs 10-fold cross-validation to partition the training and test sets and initializes the parameter matrices.

3.3 It trains a random forest to generate a similarity matrix.

3.4 It optimizes the objective function.

3.5 Based on the coefficient matrix, it selects features and constructs a feature matrix, then performs classification using Multi-kernel SVM.

3.6 It calculates TP, FP, TN, FN and computes evaluation metrics.

3.7 It aggregates the classification metrics from each fold and calculates the mean and standard deviation.
