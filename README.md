# Improved-AD-diagnosis-using-multimodal-sparse-similarity-feature-selection-and-auxiliary-data
The code of Improved Alzheimer’s disease diagnosis using multimodal sparse similarity feature selection and auxiliary data

# Abstract 
Alzheimer’s disease (AD) is a severe neurodegenerative condition that predominantly affects the elderly. It can lead to memory decline, brain atrophy, and ultimately death. The mild cognitive impairment (MCI) is a transitional stage that may be progressive (pMCI), leading to AD, or stable (sMCI). Early AD diagnosis and MCI conversion are very crucial for timely treatment, slowing disease progression, and alleviating the financial burden on patients. Multimodal neuroimaging data offers complementary information and has been widely utilized to differentiate between AD and normal controls (NC), as well as to predict MCI conversion. Recent research has indicated that the inclusion of structural information could improve the performance of MCI conversion and AD diagnosis. Moreover, the pathological characteristics of AD have been demonstrated to be advantageous for its diagnosis. However, previous works only considered different ways to construct graphs to represent the structural information of samples and lacks efficient exploitation of structural information and AD pathological characteristics. In addition, the diagnostic accuracy of classifying MCI subtypes remains constrained. In this paper, we introduce a novel approach for feature selection, targeting the identification of robust neuroimaging features from magnetic resonance imaging and positron emission tomography data. Initially, we utilize the random forest method to calculate the similarity matrix for each modality. Subsequently, we integrate auxiliary data into the objective function and similarity information using the sparse regression method for feature selection. Then, we utilize the 𝑙2,1 norm regularization term to select the shared features from predictor data and auxiliary data. Lastly, we utilize a multi-kernel support vector machine to integrate multimodal data and perform classification tasks. Extensive experiments conducted on the ADNI dataset demonstrate that our method surpasses state-of-the-art approaches, achieving classification accuracies of 80.87%, 80.36%, and 81.78% for pMCI vs. sMCI, MCI vs. NC, and MCI vs. AD, respectively.

# Code Usage Instructions
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

# Paper Link
https://www.sciencedirect.com/science/article/abs/pii/S1746809425009966
