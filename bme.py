# Data Processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scipy.stats import randint
import os
import seaborn as sns

breast_cancer = pd.read_csv("data.csv")
breast_cancer = breast_cancer.drop(columns=['id'])
print(breast_cancer.shape)
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M': 0, 'B': 1})
X = breast_cancer.iloc[:,1:31]
y = breast_cancer.iloc[:,0]

"""
if os.path.isfile("data.csv"):
    for feature in X.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=breast_cancer, x=feature, hue='diagnosis', kde=True)
        plt.title(f'{feature} Distribution')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend(['Benign', 'Malignant'])
        if not os.path.exists("graphs"):
            os.makedirs("graphs")
        plt.savefig(f'graphs/{feature}_distribution.png')
        plt.close()
    correlation_matrix = breast_cancer.corr()
    plt.figure(figsize=(30, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f'graphs/correlation_matrix.png')
    plt.close()
else:
    print("File 'data.csv' not found.")
"""
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=42)

rf_model = RandomForestClassifier(n_estimators=20, random_state=42)
rf_model.fit(Xtrain, ytrain)
rf_predictions = rf_model.predict(Xtest)

rf_accuracy = accuracy_score(ytest, rf_predictions)
print("RandomForestClassifier model:", rf_accuracy)

cm = confusion_matrix(ytest, rf_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap='Reds')
plt.title('Confusion Matrix')
plt.savefig('graphs/confusion_matrix.png')