# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:23:55 2021

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
from sklearn.ensemble import GradientBoostingClassifier

from yellowbrick.model_selection import LearningCurve

#Use Lung Med-Lat distance and Clav-Lung Sup Distance just as an example
target_combo = (0,5)
# Split data
X = X_modified[:, target_combo]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# Define your model and hyperparameters
model = LogisticRegression(max_iter = 10000, solver = 'liblinear', random_state = 0)

weights = np.loadtxt('H:/Feat_importances_txt_files/V20_Weighting.txt')
weights_exp = np.loadtxt('H:/Feat_importances_txt_files/V20_Weighting_exp.txt')

X_train, X_test, weights_train, weights_test = train_test_split(X,weights_exp,test_size=0.3,random_state=1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = model.score(X_test, y_test)
acc_train = model.score(X_train, y_train)


def plot_decision_function(classifier, sample_weight, axis, title):
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
                 cmap=plt.cm.bone, edgecolors='black')

    axis.axis('off')
    axis.set_title(title)
    
from sklearn import svm
sample_weight_constant = np.ones(len(X))

clf_weights = svm.SVC(gamma=1)
clf_weights.fit(X, y, sample_weight=weights)


clf_weights_exp = svm.SVC(gamma=1)
clf_weights_exp.fit(X, y, sample_weight=weights_exp)

clf_no_weights = svm.SVC(gamma=1)
clf_no_weights.fit(X, y)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
                       "Constant weights")
plot_decision_function(clf_weights, weights, axes[1],
                       "Modified weights")
plot_decision_function(clf_weights_exp, weights_exp, axes[2],
                       "Exp Modified weights")