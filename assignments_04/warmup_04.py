import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#ROC 01
log_reg = LogisticRegression(max_iter=1000,random_state=42)
log_reg.fit(X_train,y_train)
y_probs = log_reg.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test,y_probs)

print(f"AUC not scaled: {auc:.3f}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled,y_train)
y_probs_scaled = knn.predict_proba(X_test_scaled)[:,1]
auc_scaled = roc_auc_score(y_test,y_probs_scaled)

print(f"AUC scaled: {auc_scaled:.3f}")

#KNN has the better auc score meaning it has a better-discriminating 
# model than the logistic regression.