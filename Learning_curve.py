# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 08:54:58 2021

@author: fletcherbarrett
"""

import time

t0 = time.time()

from csv import reader
import numpy as np
from Decision_space import decision_plot
from Classifier_Importance import *
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


from collections import Counter

def get_data(filename):
    data = []
    feature_names = []
    with open(filename) as f:
        patient_names = ",".join(next(f).split()).split(',')[1:] 
        
        next(f) # throw away ----
        
        for line in f:
            feature_names.append(",".join(line.split()).split(',')[0])
            j = ",".join(line.split()).split(',')[1:]
            data.append([float(j[i]) for i in range(len(j))])
    return data, patient_names, feature_names

#Pull data from txt files written using Extract_feature.py
f1 = 'H:/Feat_importances_txt_files/Feature_Data_DIBH.txt'
f2 = 'H:/Feat_importances_txt_files/Feature_Data_FB.txt'
data1, pat_list1,feature_names = get_data(f1)
data2, pat_list2,feature_names = get_data(f2)
 
X_full = np.concatenate((np.transpose(data1),np.transpose(data2)))
y = np.array(len(pat_list1)*[0] + len(pat_list2)*[1])

#Remove some bad metrics :(
X_modified = np.delete(X_full,[8,23,24,25,26,27,28],axis=1)

# df = pd.DataFrame(X,columns=feature_names)

import numpy as np

from sklearn.model_selection import KFold

from yellowbrick.model_selection import LearningCurve

#Pass the most successful features to the learning curve:
#Lung width, Lung length, and Clavicle to Lung distance
target_combo = (0,2,5)

# Split data
X = X_modified[:, target_combo]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Create the learning curve visualizer
    #Update the random state here if you want to look at different data combos in your 5 folds
cv = KFold(n_splits = 5)

# Set your plot values
    # Update this if you want to change coarseness of dataset sizes
sizes = np.linspace(0.3, 1.0, 10)

# Define your model and hyperparameters
model = LogisticRegression(max_iter = 10000, solver = 'liblinear', random_state = 0)

# Set up visualizer
visualizer = LearningCurve(model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=-1)


visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()

t1 = time.time()

print('This code took: '+ str(t1-t0)+ 'seconds')