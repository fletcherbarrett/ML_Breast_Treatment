# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:48:09 2021

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

def model_performance(X_train, X_test, y_train, y_test, clf, model_acc, model_conf, model_sen, model_spe):
    clf.fit(X_train, y_train)
    model_acc.append(clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    model_conf.append(confusion_matrix(y_test, y_pred))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    #Sensitivity
    model_sen.append(tp/(tp+fn))
    #Specificity
    model_spe.append(tn/(tn+fp))
    
    return model_acc,model_conf,model_sen,model_spe
    

def get_best_model(X_train, X_test, y_train, y_test):
    
    #Initialize some lists to keep track of models and their performance metrics
    Models = []   
    model_acc = []
    model_conf = []
    model_sen = []
    model_spe =[]
    
    ##########################Decision Tree Classifier##########################
    from sklearn import tree    
    Models.append(tree.DecisionTreeClassifier(max_depth=5, random_state = 0))
    
    ##########################Random Forest Classifier##########################
    from sklearn import ensemble
    Models.append(ensemble.RandomForestClassifier(n_estimators=100, random_state = 0))

    ##########################Gradient Boosting Classifier##########################
    Models.append(ensemble.GradientBoostingClassifier(random_state = 0))
  
    ##########################Gradient Boosting Classifier-tuned##########################
    Models.append(ensemble.GradientBoostingClassifier(n_estimators=150, random_state = 0))
  
    ##########################AdaBoost Classifier-tuned##########################
    Models.append(ensemble.AdaBoostClassifier(random_state = 0))
    
    ##########################ExtraTrees Classifier-tuned##########################
    Models.append(ensemble.ExtraTreesClassifier(random_state = 0))

    ##########################Naive Bayes Classifier##########################
    from sklearn.naive_bayes import GaussianNB
    Models.append(GaussianNB())
    
    ##########################KNN Classifier##########################
    from sklearn.neighbors import KNeighborsClassifier
    Models.append(KNeighborsClassifier(n_neighbors = 5))

    ##########################Logistic Regression Classifier##########################
    from sklearn.linear_model import LogisticRegression
    Models.append(LogisticRegression(max_iter = 10000, solver = 'liblinear', random_state = 0))
    
    ##########################SVM Classifier##########################
    from sklearn.svm import SVC
    Models.append(SVC(probability = True, kernel = 'linear', random_state = 0))
    
    ##########################Bagging SVC Classifier##########################
    from sklearn.ensemble import BaggingClassifier    
    Models.append(BaggingClassifier(base_estimator=SVC(), random_state = 0))  
    
    ##########################Voting Classifier-tuned##########################
    from sklearn.ensemble import VotingClassifier
    Models.append(VotingClassifier(estimators=[
            ('et', Models[5]), ('rf', Models[1]), ('gnb', Models[2])], voting='hard'))

    
    from sklearn.ensemble import VotingClassifier
    Models.append(VotingClassifier(estimators=[
            ('et', Models[5]), ('rf', Models[1]), ('gnb', Models[2])], voting='soft'))
    
    for each_model in Models:
        model_acc,model_conf,model_sen,model_spe = model_performance(X_train, X_test, y_train, y_test, each_model, model_acc, model_conf, model_sen, model_spe)
    
    return model_acc,model_conf,model_sen,model_spe

#Pull data from txt files written using Extract_feature.py
f1 = 'H:/Feat_importances_txt_files/Feature_Data_DIBH.txt'
f2 = 'H:/Feat_importances_txt_files/Feature_Data_FB.txt'
data1, pat_list1,feature_names = get_data(f1)
data2, pat_list2,feature_names = get_data(f2)
 
X_full = np.concatenate((np.transpose(data1),np.transpose(data2)))
y = np.array(len(pat_list1)*[0] + len(pat_list2)*[1])

#Remove some bad metrics :(
X_modified = np.delete(X_full,[8,23,24,25,26,27,28],axis=1)

#Set the number of random variations pulled to create testing and training
random_seeds=100

all_combos = []

acc_all_combos = []
acc_all_combos_err = []

confus_all_combos = []
confus_all_combos_err = []

sen_all_combos = []
sen_all_combos_err = []

spe_all_combos = []
spe_all_combos_err = []

for feat_num in range(1,4):
    import itertools
    combo_set =list(itertools.combinations(list(range(22)), feat_num))
    
    for combo in range(len(combo_set)):
        print('We are on feature combination: '+ str(combo_set[combo]))
        all_combos.append(combo_set[combo])
        X = X_modified[:, combo_set[combo]]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        l_acc=[]
        l_con = []
        l_sen = []
        l_spe = []
        
        for seed in range(random_seeds):
            from sklearn.model_selection import train_test_split
            
            # We will be using a 30%/70% test/train
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=seed)
            
            acc,confus,sen,spe = get_best_model(X_train, X_test, y_train, y_test)
            l_acc.append(acc)
            l_con.append(confus)
            l_sen.append(sen)
            l_spe.append(spe)
        
        
        acc_all_combos.append(np.mean(l_acc,axis=0))
        acc_all_combos_err.append(np.std(l_acc,axis=0))
        
        confus_all_combos.append(np.mean(l_con,axis=0))
        confus_all_combos_err.append(np.std(l_con,axis=0))
        
        sen_all_combos.append(np.mean(l_sen,axis=0))
        sen_all_combos_err.append(np.std(l_sen,axis=0))
        
        spe_all_combos.append(np.mean(l_spe,axis=0))
        spe_all_combos_err.append(np.std(l_spe,axis=0))

#First value of index is the combination and second is the algorithm that worked
index = np.where(acc_all_combos==np.max(acc_all_combos))
#confus_all_combos[i][j] will give desired confus mat with highest accuracy score
#all_combos[i] will give the desired feature combonation that provided the highest accuracy

t1 = time.time()

print('This code took: '+ str(t1-t0)+ 'seconds')