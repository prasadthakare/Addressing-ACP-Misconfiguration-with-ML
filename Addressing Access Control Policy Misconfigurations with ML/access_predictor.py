
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:39:13 2019

@author: prasad.thakare
"""
#Import Liabraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

#Import Data
ds1 = pd.read_csv('Combined_Arranged.csv') 

#Label Encoding
ds1=ds1.apply(LabelEncoder().fit_transform)

#Specify the selected features after feature selection
X = ds1.iloc[:, 1:6].values

#Specify the resource to be classified
y = ds1.iloc[:, 35].values

#get univariate chi squared scores for feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)


#Use Hot Encoding for categorical variables

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [144])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [188])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [195])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [198])
X = onehotencoder.fit_transform(X).toarray()

#split into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


#Grid search method to determine linear or non-linear data
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 0.1, random_state = 0, probability = True)

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy')                           
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#fit the SVM classifier with an rbf kernel to the training data
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 0.1, random_state = 0, probability = True)
classifier.fit(X_train, y_train)

#get predictions
y_pred = classifier.predict(X_test)

#get prediction probabilities
y_pred_prob = classifier.predict_proba(X_test)
yprob = y_pred_prob[:, 1]

#get AUC score
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, yprob)

#plot ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, yprob)
ax = plt.subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--')
ax.plot(fpr, tpr, marker='.', label='SVM (79.95)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('AUC and ROC ENT')
ax.legend()
plt.show()

#Specify the classification threshold

y_pred_prob_1 = classifier.predict_proba(X_test)[:, 1]
y_pred_prob_1=y_pred_prob_1.reshape(1, -1)

from sklearn.preprocessing import binarize
y_pred_class = binarize(y_pred_prob_1, 0.60)[0]


#Obtain the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm1 = metrics.confusion_matrix(y_test, y_pred_class)

#obtain overall accuracy
from sklearn.metrics import accuracy_score
print("Accuracy = ",accuracy_score(y_test,y_pred))













