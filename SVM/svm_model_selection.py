# python modules
import sys
from os import path
import numpy as np
from sklearn import svm
import pickle
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
#
# my modules
sys.path.append(path.abspath('/Users/susmitadatta/Metis/Proj03/CIFAR_10/Models'))
sys.path.append(path.abspath('/Users/susmitadatta/Metis/Proj03/CIFAR_10/'))
import ml_module as mlmod
import data_transform_module as dtm

# Traning features and labels
feature_path = "~/CIFAR_10/Features/ImageFeatures/"
label_path = "~/CIFAR_10/Features/ImageData/"


ytrn = dtm.unpickleSomething(label_path, 'label_train.p')


result_path = "~/SVM/GridSearch_Results/"

# DEFINE A GRID SEARCH
Xtrn = dtm.unpickleSomething(feature_path, 'trn_pca100.p')
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 0.5, 0.1, 5, 10]}
clf = GridSearchCV(svm.SVC(), parameters)
Xtrn = dtm.unpickleSomething(feature_path, 'trn_pca100.p')
clf.fit(Xtrn, ytrn)
results = clf.cv_results_
dtm.pickleSomething(results, result_path, 'svm_pca100.p')

# scores = cross_val_score(clf, Xtrn, ytrn, cv=3)
# y_pred = cross_val_predict(clf, Xtrn, ytrn, cv=3)
# accuracy = metrics.accuracy_score(ytrn, predicted) 
# print(scores, y_pred_accuracy)


# DEFINE A GRID SEARCH
Xtrn = dtm.unpickleSomething(feature_path, 'trn_pca10.p')
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 0.5, 0.1, 5, 10]}
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 0.5, 0.1, 5, 10]}
clf = GridSearchCV(svm.SVC(), parameters)
clf.fit(Xtrn, ytrn)
results = clf.cv_results_
dtm.pickleSomething(results, result_path, 'svm_pca10.p')


                            

