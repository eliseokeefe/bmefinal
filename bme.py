# Data Processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scipy.stats import randint

#from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
#heart_disease = fetch_ucirepo(id=45) 

"""
heart_disease_two = pd.read_csv("heart.csv")
print(heart_disease_two)
X = heart_disease_two.iloc[:,0:13]
y = heart_disease_two.iloc[:,13]
"""

"""
liver_disease = pd.read_csv("indian_liver_patient.csv")
liver_disease = liver_disease.dropna()
liver_disease['Gender'] = liver_disease['Gender'].map({'Male': 0, 'Female': 1})
X = liver_disease.iloc[:,0:10]
y = liver_disease.iloc[:,10]
"""

"""
heart_disease_three = pd.read_csv("heart 2.csv")
heart_disease_three = heart_disease_three.dropna()
heart_disease_three['Sex'] = heart_disease_three['Sex'].map({'M': 0, 'F': 1})
heart_disease_three['ChestPainType'] = heart_disease_three['ChestPainType'].map({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3})
heart_disease_three['RestingECG'] = heart_disease_three['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
heart_disease_three['ExerciseAngina'] = heart_disease_three['ExerciseAngina'].map({'N': 0, 'Y': 1})
heart_disease_three['ST_Slope'] = heart_disease_three['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

X = heart_disease_three.iloc[:,0:11]
y = heart_disease_three.iloc[:,11]
"""

breast_cancer = pd.read_csv("data.csv")
breast_cancer = breast_cancer.drop(columns=['id'])
print(breast_cancer.shape)
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M': 0, 'B': 1})
X = breast_cancer.iloc[:,1:31]
y = breast_cancer.iloc[:,0]

import seaborn as sns

y = breast_cancer.iloc[:,0]

# Create graphs of every feature

for feature in X.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=breast_cancer, x=feature, hue='diagnosis', kde=True)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(['Benign', 'Malignant'])
    plt.show()


# Create correlation matrix
correlation_matrix = breast_cancer.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=42)

rf_model = RandomForestClassifier(n_estimators=20, random_state=42)
rf_model.fit(Xtrain, ytrain)
rf_predictions = rf_model.predict(Xtest)

rf_accuracy = accuracy_score(ytest, rf_predictions)
print("RandomForestClassifier model:", rf_accuracy)

# Confusion Matrix
cm = confusion_matrix(ytest, rf_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap='Reds')
plt.title('Confusion Matrix')
plt.show()

"""
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

heart_disease_data = X.join(y)
heart_disease_data = heart_disease_data.dropna()

print(heart_disease_data.shape)

X = heart_disease_data.iloc[:,0:13]
y = heart_disease_data.iloc[:,13]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
clf = LogisticRegression(random_state=42, solver='sag', max_iter=10000).fit(Xtrain, ytrain)

print(clf.score(Xtest, ytest))


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
