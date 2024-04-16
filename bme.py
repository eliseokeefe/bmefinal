# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
print(heart_disease.data.features.dtypes)
X = heart_disease.data.features 
y = heart_disease.data.targets 



"""
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=42)
rf_model = RandomForestClassifier(n_estimators=20, random_state=42)
rf_model.fit(Xtrain, ytrain)
rf_predictions = rf_model.predict(Xtest)
rf_accuracy = accuracy_score(ytest, rf_predictions)
print("RandomForestClassifier model:", rf_accuracy)
"""
  
# metadata 
# print(heart_disease.metadata) 
  
# variable information 
# print(heart_disease.variables) 
