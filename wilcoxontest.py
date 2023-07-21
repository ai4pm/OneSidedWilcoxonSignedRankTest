# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:43:56 2023

@author: TEENA SHARMA
"""

## This code is written to provide the step by step implmentation of performing
## One sided Wilcoxon Signed Rank Test to analysis the statistical significance
## of the results obtained.
## In this code, I am assuming that a sample machine learning task result is already
## available with 20 iterations for the following:
## 1. Mixture 0, Mixture 1, Mixture 2
## 2. Independent 1, Independent 2
## 3. Naive Transfer
## 4. Transfer Learning - Supervised, Unsupervised, CCSA
## We want to analyze the statistical significance of the following:
## 1. Mixture-Gap
## 2. Independent-Gap
## 3. Transfer-Superior

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# read the result file
res_file = pd.read_excel('TCGA-KIPAN-BLACK-WHITE-Protein-DSS-4YR.xlsx')
# remove the columns which are not required for convenience
res_file = res_file.drop(columns=['Unnamed: 0'])
# p-value for statistical significance threshold
p_TH = 0.05
# keep only transfer learning columns in the result and removing the rest
res_TL = res_file.drop(columns=['Mixture 0','Mixture 1','Mixture 2',
                                'Independent 1','Independent 2',
                                'Naive Transfer'])
# find average of all individual transfer learning methods
avgVal = res_TL.mean(axis=0)
# identify the transfer learning method with highest average value
best_TL_index = avgVal.idxmax()
# average value of respective machine learning schemes
ML0 = res_file['Mixture 0']
ML1 = res_file['Mixture 1']
ML2 = res_file['Mixture 2']
IL1 = res_file['Independent 1']
IL2 = res_file['Independent 2']
NT = res_file['Naive Transfer']
TL = res_file[best_TL_index] # here, transfer learning with highest performance
# performing one sided wilcoxon signed rank test...
# NOTE: In next two lines of code, one sided wilcoxon signed rank test is performed assuming 
# that Mixture 1 is higher than Mixture 2 and Independent 1 is higher than Independent 2
w1,ML1_ML2_pvalue = wilcoxon(ML1,ML2,alternative="greater",mode="approx")
w2,IL1_IL2_pvalue = wilcoxon(IL1,IL2,alternative="greater",mode="approx")
# NOTE: For transfer learning, we perform one sided wilcoxon signed rank test after identifying
# that average value of transfer learning method is higher or lower than Mixture 2 and Independent 2
if (IL2.mean(axis=0)-TL.mean(axis=0))>=0: # if Independent 2 >= transfer learning
    w3,IL2_TL_pvalue = wilcoxon(IL2,TL,alternative="greater",mode="approx")
elif (IL2.mean(axis=0)-TL.mean(axis=0))<=0: # if Independent 2 <= transfer learning
    w3,IL2_TL_pvalue = wilcoxon(IL2,TL,alternative="less",mode="approx")
if (ML2.mean(axis=0)-TL.mean(axis=0))>=0: # if Mixture 2 >= transfer learning
    w4,ML2_TL_pvalue = wilcoxon(ML2,TL,alternative="greater",mode="approx")
elif (ML2.mean(axis=0)-TL.mean(axis=0))<=0: # if Mixture 2 <= transfer learning
    w4,ML2_TL_pvalue = wilcoxon(ML2,TL,alternative="less",mode="approx")
# For convenience, in next lines of code, I am obtaining the performance pattern category for this task
# e.g. Pattern 000 / Pattern 001 / Pattern 010 / .... and so on
t1 = 1 if ML1_ML2_pvalue<p_TH else 0
t2 = 1 if IL1_IL2_pvalue<p_TH else 0
t3_ML2_IL2_TL = 1 if (IL2_TL_pvalue<p_TH and ML2_TL_pvalue<p_TH) else 0
combName_ML2_IL2_TL = 'Pattern '+str(t1)+str(t2)+str(t3_ML2_IL2_TL)
print('The performance pattern category for this machine learning task is: '+combName_ML2_IL2_TL)

    

